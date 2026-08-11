"""
Centralized configuration for the Zero-Shot Brand Content Generator.

Follows the dataclass-based config style used in week 8/10's ``config.py`` —
a single place to change model choice, sampling defaults, and injection-
defense settings without hunting through app.py/src/ for magic numbers.

HIGHLIGHTS: ADR 0006 অনুযায়ী ("Standardize on Ollama + llama3.1:8b"), model
নামটা এখানে *data* হিসেবে রাখা হয়েছে, src/llm/client.py, app.py, আর tests-এ
ছড়িয়ে থাকা literal string হিসেবে না। কোনো student-এর মেশিন RAM-constrained
হলে ``llama3.2:3b`` fallback দরকার হতে পারে (নিচে LLMConfig.fallback_model
দেখুন) — সেটা এখানে ONE-LINE change, পুরো codebase জুড়ে grep-and-replace না।
Week 10-এর EmbeddingConfig.model_name-এর মতোই একই যুক্তি।
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LLMConfig:
    """Everything about *which* local LLM we call and how we sample from it.

    HIGHLIGHTS: ``model`` আর ``fallback_model`` — এই দুটোই একমাত্র জিনিস যা
    এই project-কে একটা নির্দিষ্ট Ollama model-এর সাথে couple করে।
    src/llm/client.py এই value গুলো lazily পড়ে (import time-এ না, function-এর
    ভেতরে, call time-এ) — import করার সময়ই Ollama-র সাথে connect হয়ে গেলে
    সমস্যা হবে; পুরো "কেন lazy" যুক্তির জন্য সেই file-এর module docstring দেখুন
    (week 10-এর lazy SentenceTransformer loading-এর মতোই, শুধু downloaded
    model file-এর বদলে একটা local LLM server-এর জন্য প্রয়োগ করা হয়েছে)।
    """

    # llama3.1:8b: the course-wide default per docs/adr/0006. Good
    # instruction-following for a "write on-brand marketing copy" task,
    # comfortable on a 16GB-RAM machine. See src/llm/client.py's HIGHLIGHTS
    # for the full Model Reference Table (adapted from the Week 11 lecture,
    # section 4.1) explaining the tradeoffs against the alternatives below.
    model: str = "llama3.1:8b"

    # llama3.2:3b: documented fallback for students on the course's stated
    # minimum spec (8GB RAM) where an 8B model quantized via Ollama's
    # default Q4 is workable on 16GB but tight on 8GB (ADR 0006). Swapping
    # to this is a config change, not a code change — see README "Quick
    # Start" for how to select it.
    fallback_model: str = "llama3.2:3b"

    # Sampling defaults. Per the lecture's section 3.4, marketing copy
    # typically wants a MEDIUM temperature (0.7-0.9): creative enough to
    # not sound robotic, but not so high it drifts off brand voice. These
    # are just the slider *starting positions* in app.py — the user can
    # move them; nothing here is a hard limit.
    default_temperature: float = 0.8
    default_top_p: float = 0.9

    # Ollama's request timeout. Local inference on CPU for an 8B model can
    # legitimately take tens of seconds for a longer CoT-wrapped generation
    # — this is generous on purpose rather than tuned to be tight, because
    # a premature timeout on a slow laptop is a worse student experience
    # than an occasional long wait.
    timeout_seconds: int = 120


@dataclass(frozen=True)
class InjectionDefenseConfig:
    """Knobs for the prompt-injection defense pipeline (src/security/).

    HIGHLIGHTS: delimiter tag আর "suspicious phrase" list — দুটোই এখানে DATA
    হিসেবে রাখা হয়েছে, src/security/injection_guard.py-এর ভেতরে পোঁতা string
    literal হিসেবে না। এর দুইটা কারণ আছে:
      ১. tests exact delimiter tag-এর ওপর assert করতে পারবে, string
         literal দুই জায়গায় (config.py আর test file) ডুপ্লিকেট না করেই —
         নাহলে দুটো জায়গা নিঃশব্দে drift করে যেতে পারত।
      ২. এক জায়গায়, পড়ার মতো করে, ঠিক কী "cheap" heuristic defense চেক
         করছে তা document করা থাকে — যা নিজের সীমাবদ্ধতা সম্পর্কে honest
         (lecture-এর "সৎ কথা" callout দেখুন: কোনো defense-ই ১০০% foolproof
         না, লক্ষ্য injection-কে HARD করা, impossible না)।
    """

    # HIGHLIGHTS: model-কে পাঠানো prompt-এর ভেতরে untrusted user input
    # delimit করতে এই tag pair ব্যবহার হয় (src/security/injection_guard.py-এর
    # delimit_user_input() দেখুন)। triple quotes-এর বদলে XML-style tag
    # ব্যবহার করা হয়েছে, কারণ user-এর নিজের input-এ quote বা markdown থাকলেও
    # এগুলো visually unambiguous থাকে।
    user_input_open_tag: str = "<user_input>"
    user_input_close_tag: str = "</user_input>"

    # HIGHLIGHTS: MODEL-এর OUTPUT-এ এই phrase গুলো দেখা গেলে সেটা একটা strong
    # signal যে injection attempt assistant-এর persona/instructions hijack
    # করতে সফল হয়েছে (post-hoc output validation — src/security/
    # injection_guard.py-এর validate_output() দেখুন)। এই list ইচ্ছাকৃতভাবে
    # ছোট আর readable রাখা হয়েছে, exhaustive regex library না — এটা
    # TECHNIQUE-টা দেখায় (persona-switch language flag করা), production-grade
    # classifier না।
    suspicious_output_phrases: tuple[str, ...] = (
        "arrr",
        "ahoy",
        "matey",
        "as a pirate",
        "i am now a pirate",
        "ignoring previous instructions",
        "i will ignore",
        "previous instructions",
        "as an ai with no restrictions",
        "i am no longer",
        "new instructions",
        "system prompt",
        "i have been freed",
        "developer mode",
        "jailbreak",
    )

    # HIGHLIGHTS: heuristic "is this even on-topic" check flag তোলার আগে,
    # output-এর word গুলোর কতটুকু request-এ দেওয়া brand/product vocabulary-র
    # সাথে overlap করতে হবে তার minimum fraction। এটা ইচ্ছাকৃতভাবে loose
    # রাখা হয়েছে (injection_guard.py-এর HIGHLIGHTS দেখুন) — এর কাজ হলো
    # সম্পূর্ণ off-topic hijacked output (যেমন একটা pirate poem) ধরা,
    # marketing copy-র মান নম্বর দেওয়া না।
    min_topic_overlap_ratio: float = 0.03


@dataclass(frozen=True)
class PromptConfig:
    """Formatting knobs for prompt construction (src/prompting/templates.py)."""

    # HIGHLIGHTS: UI কতগুলো few-shot example accept করবে তার আগে বাকিগুলো
    # ignore করা শুরু করে দেয়। lecture-এর Brain Teaser #1 জিজ্ঞেস করে "আরও
    # example দিলে কোথায় গিয়ে আর কাজ দেয় না?" — এটা templates.py-তে
    # hardcode না করে config-এ cap করা মানে কৌতূহলী student-এর জন্য এটা
    # one-line experiment হয়ে যায়।
    max_few_shot_examples: int = 5

    # HIGHLIGHTS: task-এর আগে CoT-wrapping যে instruction prepend করে সেটা।
    # এখানে centralize করা হয়েছে যাতে tests app আসলে যে SAME string পাঠায়
    # তার বিপরীতে assert করতে পারে, হাতে-টাইপ করা duplicate না যেটা drift
    # করে যেতে পারে।
    cot_instruction: str = (
        "Before writing the final copy, briefly reason step by step: "
        "(1) who is the target audience, (2) what pain point or desire "
        "does this product address, (3) which brand-voice traits are most "
        "relevant here. Label this section 'Reasoning:'. Then write the "
        "final copy under a line that says 'Final Copy:'."
    )


@dataclass(frozen=True)
class AppConfig:
    """Streamlit page-level settings."""

    page_title: str = "Zero-Shot Brand Content Generator"
    page_icon: str = "🪄"
    layout: str = "wide"


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all sub-configs.

    HIGHLIGHTS: একটা বড় flat dataclass-এর বদলে কয়েকটা ছোট ছোট frozen
    dataclass compose করা — এটা week 8/10-এর Config pattern-এরই প্রতিফলন —
    প্রতিটা concern (LLM connection/sampling, injection defense, prompt
    formatting, app UI) আলাদাভাবে পড়া, test করা, আর যুক্তি করা যায়।
    সবজায়গায় ``frozen=True`` মানে config-কে পুরো process lifetime জুড়ে একটা
    constant হিসেবে treat করা হয়; ``config.llm.model`` ভুলবশত mid-run
    mutate হয়ে গেলে সেটা একটা bug — আমরা চাই সেটা সাথে সাথে ধরা পড়ুক
    (FrozenInstanceError), পরে inconsistent behavior থেকে debug করতে না হোক।
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    injection_defense: InjectionDefenseConfig = field(default_factory=InjectionDefenseConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    app: AppConfig = field(default_factory=AppConfig)


_config_instance: Config | None = None


def get_config() -> Config:
    """Get or create the process-wide singleton Config instance.

    HIGHLIGHTS: week 8/10-এর ``get_config()``-এর মতোই একই singleton pattern —
    একটা module-level cache মানে প্রতিটা caller (app.py, src/llm/client.py,
    src/prompting/templates.py, src/security/injection_guard.py, tests)
    ঠিক একই config object দেখে। এটা বানানো সস্তা (শুধু dataclass construction,
    কোনো I/O না, network না), তাই YAML file থেকে load না করলেও এই pattern
    পুনরায় ব্যবহার করায় কোনো meaningful cost নেই।
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
