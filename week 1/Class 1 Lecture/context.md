# Week 1 — Class 1: Data Foundations (Setup & Syntax)

---

## 1. AI Engineer vs. ML Engineer: Roles & Responsibilities

### Topic
The distinction between an **AI Engineer** and an **ML Engineer**, and why both roles are essential in the modern AI product lifecycle.

### Why It Is Related
Before writing a single line of code, you must know *which hat* you are wearing. In industry, job descriptions often conflate these titles, but the skill sets diverge significantly. An AI Engineering course must ground you in the reality of the job market and help you choose your trajectory.

### How It Works
- **ML Engineer**: Focuses on the *training pipeline*. They handle data collection, feature engineering, model selection, hyperparameter tuning, distributed training, and model versioning. Their output is a trained artifact (a `.pt`, `.pkl`, or `.onnx` file).
- **AI Engineer**: Focuses on the *inference pipeline* and *product integration*. They take pre-trained models (open-source or API-based), optimize them for latency/cost, build RAG systems, design prompt templates, handle API orchestration, and deploy end-user applications. Their output is a *working product*.

### Detailed Brief: How It Works & Valid Points
| Dimension | ML Engineer | AI Engineer |
|-----------|-------------|-------------|
| **Primary Question** | "How do I make the model accurate?" | "How do I make the model useful?" |
| **Core Tools** | PyTorch, JAX, Kubeflow, Weights & Biases | LangChain, Ollama, FastAPI, Docker, Redis |
| **Data Focus** | Training datasets, embeddings, vectors | Prompts, context windows, user sessions |
| **Deployment** | Batch inference, model serving (TorchServe) | Real-time APIs, streaming, edge devices |

**Valid Point**: The rise of LLMs has shifted the industry. It is now cheaper to *fine-tune* or *prompt* a foundation model than to train one from scratch. This shifted demand from ML Engineers (who build models) to AI Engineers (who wire models into products). Understanding this boundary prevents you from over-engineering solutions.

### Analogy
Think of a **car factory**:
- The **ML Engineer** is the *powertrain designer* — they design the engine, test fuel mixtures, and optimize horsepower. They work in the lab.
- The **AI Engineer** is the *race car driver + pit crew chief* — they don't build the engine, but they know exactly how to tune the suspension, choose the tires for the weather, and drive the car to win the race. They work on the track.

---

## 2. Environment Setup: Python venv, Conda, VS Code

### Topic
Setting up a reproducible, isolated development environment using Python's built-in `venv`, Anaconda/Miniconda, and configuring VS Code as the IDE.

### Why It Is Related
AI Engineering is dependency hell. One project needs PyTorch 2.1 with CUDA 12.1; another needs TensorFlow 2.15 with CUDA 11.8. Without isolation, your global Python installation becomes a graveyard of conflicting packages. Reproducibility is the difference between "it works on my machine" and production deployment.

### How It Works
1. **`venv` (Standard Library)**: Creates a lightweight virtual environment by copying the Python binary and maintaining a separate `site-packages` directory. It uses `pip` for package management.
2. **Conda**: A cross-language package manager and environment manager. It installs binary packages (including non-Python dependencies like CUDA toolkit, MKL, or system libraries) without requiring compilers.
3. **VS Code**: A lightweight editor with extensions for Python linting (Pylance), Jupyter notebooks, Docker, and remote development. The `.vscode/settings.json` file enforces consistency across team members.

### Detailed Brief: How It Works & Valid Points
- **venv workflow**: `python -m venv .venv` → `source .venv/bin/activate` → `pip install -r requirements.txt`. Best for *deployment* environments because it is standard and has zero external dependencies.
- **Conda workflow**: `conda create -n aieng python=3.11` → `conda activate aieng` → `conda env export > environment.yml`. Best for *data science* workflows because it handles C-library dependencies (e.g., `cudatoolkit`) that pip cannot.
- **VS Code Configuration**: The `.vscode/` directory stores `settings.json` (formatter, linter, Python interpreter path) and `launch.json` (debug profiles). This turns a personal editor into a team-wide standardized workstation.

**Valid Point**: In production, Docker containers are built from `requirements.txt`, not `conda`. Learning `venv` + `pip` first makes you deployment-ready. Conda is excellent for local experimentation but adds bloat to CI/CD pipelines.

### Analogy
Imagine you are a **chef** with multiple restaurants:
- **`venv`** is like having separate *prep stations* in one shared kitchen. Each station has its own knives and ingredients. Cheap, fast, but still in the same building.
- **Conda** is like having separate *restaurant branches* with fully stocked pantries, including imported spices you cannot find at a regular grocery store. Heavier, but self-contained.
- **VS Code** is your *universal recipe book and kitchen layout* — no matter which restaurant you walk into, the stove is in the same place.

---

## 3. Local LLM Setup with Ollama

### Topic
Running Large Language Models (LLMs) locally using **Ollama**, a lightweight framework that abstracts model downloading, GPU/CPU inference, and API serving.

### Why It Is Related
Cloud LLM APIs (OpenAI, Anthropic) charge per token and send your data to third-party servers. For prototyping, sensitive data, or cost control, local inference is non-negotiable. Ollama reduces the barrier from "I want to run LLaMA" to "it is running" from hours to minutes.

### How It Works
1. **Ollama Engine**: A Go-binary background service that manages model weights (GGUF format), allocates memory, and exposes a REST API at `localhost:11434`.
2. **Model Pulling**: Ollama downloads quantized models (e.g., `llama3.2`, `mistral`, `qwen2.5`) from its registry. Quantization (Q4_K_M, Q5_K_M) reduces model size by representing weights with fewer bits, trading marginal accuracy for massive speed gains.
3. **Inference**: You send a POST request with `{"model": "...", "prompt": "..."}` and receive a streamed or batched response. Ollama handles tokenization, KV-cache management, and context window truncation internally.

### Detailed Brief: How It Works & Valid Points
- **GGUF Format**: A binary format optimized for CPU inference via `llama.cpp`. It stores tensors in a memory-mappable layout, allowing the OS to load only the necessary pages into RAM rather than the entire model.
- **Quantization Impact**: A 70B parameter model at FP16 needs ~140GB VRAM. At Q4_K_M, it needs ~42GB. At Q4_0, it needs ~38GB. For local development on a laptop with 16GB RAM, a 7B Q4 model (~4GB) is the sweet spot.
- **API Compatibility**: Ollama's API mimics OpenAI's `/v1/chat/completions` structure (via `ollama serve` or tools like `litellm`), meaning you can prototype locally and swap to GPT-4 in production with zero code changes.

**Valid Point**: Local LLMs are not just for privacy. They are for *iteration velocity*. When you are debugging a prompt or RAG pipeline, waiting 500ms for a local response beats waiting 5 seconds for an API round-trip. You can run 100 experiments in the time it takes to run 10 via cloud.

### Analogy
Think of Ollama as a **portable generator**:
- The **power grid** (OpenAI API) is reliable and powerful, but you pay per use, and if the grid goes down (rate limits, outages), you are stuck.
- **Ollama** is a diesel generator in your garage. It is not as powerful as the grid, but it is *yours*. You can flip the switch anytime, run it during a storm, and you never get a bill at the end of the month.

---

## 4. Python Data Types and Control Flow

### Topic
A rapid, engineering-focused review of Python primitives (`int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`), mutability, and control structures (`if/elif/else`, `for`, `while`, `try/except`).

### Why It Is Related
Python is the lingua franca of AI. Every framework — PyTorch, Hugging Face, LangChain, FastAPI — is built on these primitives. A shaky foundation here means you will struggle to read documentation, debug type errors, or optimize data pipelines. In AI Engineering, you manipulate *massive* dictionaries of model outputs, batch lists of prompts, and handle nested JSON configurations.

### How It Works
- **Mutability**: `list` and `dict` are mutable (in-place changes). `tuple`, `str`, `int` are immutable (reassignment creates new objects). This matters because passing a `dict` into a function modifies the original unless you explicitly `.copy()` it.
- **Control Flow**: `for` loops iterate over iterables. `while` loops run until a condition breaks. `try/except` catches exceptions, preventing crashes when an API returns malformed data.
- **Truthiness**: Empty collections (`[]`, `{}`, `''`) evaluate to `False`. Non-empty collections evaluate to `True`. This enables Pythonic patterns like `if data:` instead of `if len(data) > 0:`.

### Detailed Brief: How It Works & Valid Points
- **Dictionary as the Universal Container**: In AI Engineering, 90% of data flows as dictionaries. An LLM API response is a dict. A model config is a dict. A token log-probability mapping is a dict. Mastering dict methods (`.get()`, `.items()`, `.update()`, dict unpacking `**`) is non-negotiable.
- **List as the Batch Container**: When you send prompts to an LLM, you send a *list* of messages. When you receive embeddings, you receive a *list* of vectors. List operations (slicing, appending, extending) are your assembly line.
- **Tuple for Integrity**: Use tuples for fixed records (e.g., `(model_name, quantization_level, file_path)`). Because they are immutable, they can be used as dictionary keys or stored in sets — impossible with lists.

**Valid Point**: Python's dynamic typing is a double-edged sword. It lets you prototype fast, but it also lets you pass a `str` where an `int` should be, crashing a training job 6 hours in. Using type hints (`def foo(x: int) -> str:`) and tools like `mypy` turns Python from a scripting language into a robust engineering language.

### Analogy
Think of Python data types as **kitchen containers**:
- A **`list`** is a *stack of plates* — you can add, remove, and reorder plates freely. It is flexible but fragile (if you drop the stack, order matters).
- A **`dict`** is a *spice rack with labeled jars* — you do not care about the order; you grab "paprika" by name. It is how you organize complexity.
- A **`tuple`** is a *sealed meal-prep container* — once you pack Monday's lunch, you do not open it to swap the chicken for tofu. It guarantees consistency.
- **`if/else`** is your *recipe decision tree* — "If the chicken is 165°F, serve it. Else, keep cooking."

---

## Class 1 Achievement Map
| Achievement | Knowledge Applied |
|-------------|-------------------|
| Run a LLM model locally | Ollama installation, model pulling, REST API interaction |
| Automate a text-sorting task | Python `list`, `dict`, file I/O, control flow, string methods |

---

*End of Class 1 Lecture Material*
