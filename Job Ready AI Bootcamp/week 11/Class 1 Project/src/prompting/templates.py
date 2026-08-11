"""
Prompt construction for the Brand Content Generator: zero-shot, few-shot,
and Chain-of-Thought (CoT) wrapping — the three prompting techniques from
the Week 11 lecture, sections 3.1-3.2.

HIGHLIGHTS — prompt *text assembly* কেন src/llm/client.py থেকে আলাদা, নিজের
module-এ থাকে? Prompt string বানানো (কোন শব্দ কোন order-এ যাবে) pure string
logic, Ollama বা কোনো network call-এর সাথে কোনো dependency নেই — কিছু mock
না করেই trivially unit-test করা যায়। client.py-এর থেকে আলাদা রাখলে (যার
সত্যিই mocking দরকার, ওই module-এর docstring দেখুন) এই project-এর বেশিরভাগ
test coverage (prompt structure, few-shot formatting, CoT wording) plain,
fast, dependency-free test হিসেবে চলতে পারে — tests/test_pipeline.py দেখুন।

HIGHLIGHTS — এই module কখনো src.security.injection_guard সরাসরি call করে
না কেন? Prompt *content* construction (zero-shot বনাম few-shot বনাম CoT) আর
injection *defense* (delimiting, instruction-hierarchy framing) দুটো
আলাদা concern, যেগুলো শুধু কাকতালীয়ভাবে একই final prompt string-কে touch করে।
এই module TASK content বানায়; app.py দায়িত্ব নেয় কখন user-supplied text-কে
আগে src.security.injection_guard.delimit_user_input()-এর মধ্য দিয়ে চালাতে
হবে (নির্দিষ্টভাবে "Try to Break It" tab-এ) সেই সিদ্ধান্ত নেওয়ার। সেই
সিদ্ধান্তটা app.py-তে রাখা — এই module সবকিছু নিঃশব্দে delimit করে ফেলার
বদলে — security boundary-কে call site-এ একটা দৃশ্যমান, ইচ্ছাকৃত পছন্দ বানায়,
পাঠক মিস করে যেতে পারে এমন একটা লুকানো implementation detail না।
"""

from __future__ import annotations

from config import get_config


def build_zero_shot_prompt(brand_name: str, product_info: str, brand_voice: str) -> str:
    """Build a zero-shot task prompt: describe the brand voice in words,
    no examples.

    HIGHLIGHTS: lecture-এর section 3.1 আর 4.2 অনুযায়ী, zero-shot মডেলকে
    শুধু desired style-এর একটা DESCRIPTION থেকে কাজ করতে বলে ("friendly and
    playful") — লিখতে দ্রুত, কিন্তু মডেলের কাছে "friendly and playful"-কে
    brand-এর আসল sound থেকে আলাদাভাবে interpret করার অনেক জায়গা থেকে যায়।
    এটাই এই project-এর Few-Shot টগলের সাথে তুলনা করার জন্য ইচ্ছাকৃত baseline।
    """
    return (
        f"Write a short piece of marketing copy for the brand "
        f"\"{brand_name}\".\n\n"
        f"Product/offer details: {product_info.strip()}\n\n"
        f"Brand voice: {brand_voice.strip()}\n\n"
        f"Write copy that matches this brand voice description as closely "
        f"as possible."
    )


def build_few_shot_prompt(
    brand_name: str,
    product_info: str,
    brand_voice: str,
    examples: list[str],
) -> str:
    """Build a few-shot task prompt: 2-3 real brand-voice snippets, then
    the new task.

    Args:
        examples: 2-3 short strings of EXISTING brand copy the model
            should infer the style from (not literally reused).

    HIGHLIGHTS — example গুলো task instruction থেকে আলাদাভাবে delimit করা
    হয় কেন (প্রতিটা নিজের numbered line-এ, নিজের section-এর ভেতরে), শুধু
    এক প্যারায় instructions-এর সাথে মিশিয়ে ঢেলে দেওয়ার বদলে?
    এখানে structure আসলেই কাজ করে, শুধু readability না: "Example 1: ...
    Example 2: ... Now write a NEW ad for: ..." পড়া একটা মডেল স্পষ্ট বুঝতে
    পারে "এগুলো pattern-match করার জন্য sample" আর "এটাই আসল task" — কারণ
    numbering আর explicit "Now write a new ad in this exact style for:"
    transition (lecture-এর section 3.1 example প্রায় হুবহু) সীমানাটা মার্ক
    করে দেয়। example আর instructions এক প্যারায় মিশে গেলে, মডেলের একটা
    example-এর CONTENT-কে (নির্দিষ্ট product, নির্দিষ্ট claim) শুধু style
    reference না ভেবে আসল task-এর অংশ ভেবে নেওয়ার সম্ভাবনা বাড়ে — ঠিক
    lecture-এর "showing beats describing" পয়েন্ট (4.2) যে ambiguity এড়াতে
    চায়।

    HIGHLIGHTS — ব্যবহৃত example-এর সংখ্যা cap করা কেন
    (``config.prompt.max_few_shot_examples``)? সরাসরি lecture-এর Brain
    Teaser #1-এর সাথে জড়িত: বেশি example অনির্দিষ্টকাল সাহায্য করে না, একটা
    পয়েন্টের পর সেগুলো শুধু context পোড়ায় আর (একটা ছোট local model-এ) যে
    pattern reinforce করার কথা ছিল সেটাকেই dilute করা শুরু করে। UI সবসময়
    যুক্তিসঙ্গত সংখ্যা পাঠাবে ভরসা না করে এখানে cap করলে এই function যেখান
    থেকেই call হোক না কেন limit enforced থাকে।
    """
    cfg = get_config().prompt
    capped_examples = [e.strip() for e in examples if e.strip()][: cfg.max_few_shot_examples]

    examples_block = "\n".join(
        f"Example {i}: \"{example}\"" for i, example in enumerate(capped_examples, start=1)
    )

    return (
        f"Here are real examples of \"{brand_name}\"'s existing brand voice:\n\n"
        f"{examples_block}\n\n"
        f"Now write a NEW piece of marketing copy in this exact style for the "
        f"following product/offer — do not reuse the examples' wording, "
        f"match their tone, rhythm, and word choice instead:\n\n"
        f"Product/offer details: {product_info.strip()}\n\n"
        f"Brand voice (for reference, in addition to the examples above): "
        f"{brand_voice.strip()}"
    )


def wrap_with_chain_of_thought(task_prompt: str) -> str:
    """Prepend a step-by-step reasoning instruction to a task prompt.

    HIGHLIGHTS: lecture-এর section 3.2 অনুযায়ী, এটা কাজ করে কারণ একটা LLM
    token-by-token জেনারেট করে, প্রতিটা token এখন পর্যন্ত জেনারেট হওয়া সবকিছুর
    ওপর conditioned — মডেলকে আগে relevant context (কে audience, কী pain
    point, কোন brand-voice trait এখানে প্রাসঙ্গিক) WRITE OUT করতে বাধ্য
    করলে সেই context মডেলের নিজের context window-এ থেকে যায় যখন সে final
    copy জেনারেট করে — সরাসরি output-এ ঝাঁপ দেওয়ার চেয়ে measurably বেশি
    চিন্তাশীল উত্তরের দিকে ঠেলে দেয়। exact instruction text
    ``config.prompt.cot_instruction``-এ থাকে (এখানে hardcode না), যাতে
    tests/test_pipeline.py এই function আসলে যে SAME string ব্যবহার করে
    তার বিপরীতে assert করতে পারে, দুটো আলাদা হয়ে drift করার কোনো ঝুঁকি ছাড়াই।

    HIGHLIGHTS — শুধু "think step by step" বলার বদলে মডেলকে section
    "Reasoning:" আর "Final Copy:" লেবেল করতে বলা হয় কেন? একটা vague "think
    step by step" instruction reasoning আর final answer-কে response-এ
    মিশিয়ে ফেলে, যা app.py-এর "Show reasoning (Chain-of-Thought)"
    checkbox-এর জন্য শুধু reasoning অংশটা পরিষ্কারভাবে show/hide করা কঠিন
    করে তোলে। Explicit label app.py-কে split করার জন্য একটা সহজ, নির্ভরযোগ্য
    string দেয় (app.py-এর response-rendering logic দেখুন), দুটোকে আলাদা
    করতে দ্বিতীয় একটা LLM call বা fragile regex লাগে না।
    """
    cfg = get_config().prompt
    return f"{cfg.cot_instruction}\n\n---\n\n{task_prompt}"


def build_brand_system_prompt(brand_name: str) -> str:
    """Build the base (undefended) brand-voice system prompt.

    HIGHLIGHTS: এটা ইচ্ছাকৃতভাবে "undefended" system prompt — শুধু
    brand-copywriter role, কোনো instruction-hierarchy framing নেই। এটা নিজের
    আলাদা function হিসেবে আছে যাতে app.py-এর "Try to Break It" tab
    student-কে DIFFERENCE দেখাতে পারে: এই raw prompt একা, বনাম model-এ
    পাঠানোর আগে src.security.injection_guard.build_defended_system_prompt()
    দিয়ে চালানো একই prompt। দুটো পাশাপাশি দেখাই defense-এর effect-কে শুধু
    দাবি না করে legible করে তোলে।
    """
    return (
        f"You are a professional marketing copywriter working exclusively "
        f"for the brand \"{brand_name}\". Your only job is to write "
        f"on-brand marketing copy based on the product/offer details and "
        f"brand voice guidance you are given. Stay in this role at all times."
    )
