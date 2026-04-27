# Week 1 — Class 2: Advanced Python for AI

---

## 1. List Comprehensions

### Topic
A Pythonic syntax for creating lists by transforming and filtering iterables in a single, readable expression.

### Why It Is Related
In AI Engineering, data is a list of something: a list of prompts, a list of tokens, a list of embeddings, a list of API responses. You will transform these lists constantly — batching, filtering, normalizing. List comprehensions are the standard tool because they are fast, concise, and universally understood by Python AI practitioners.

### How It Works
A list comprehension has the form: `[expression for item in iterable if condition]`
- **`for item in iterable`**: The loop driver.
- **`expression`**: What to produce for each item (can transform, calculate, or nest).
- **`if condition`** (optional): A filter; only items passing the test are included.

Python executes the comprehension in C-speed internally, making it significantly faster than an equivalent `for` loop with `.append()` in pure Python.

### Detailed Brief: How It Works & Valid Points
- **Nested Comprehensions**: You can nest them for matrix operations or flattening nested JSON. `[token for sentence in paragraphs for token in sentence]` flattens a 2D list into 1D — common when processing tokenized text.
- **Dict & Set Comprehensions**: `{k: v for k, v in data.items() if v > 0}` creates dictionaries on the fly. In AI, you use these to build vocabulary mappings or filter feature dictionaries.
- **Generator Expressions**: `(x for x in data if x)` uses parentheses instead of brackets. It is lazy — it yields one item at a time, saving memory when processing gigabyte-scale datasets.

**Valid Point**: Readability matters more than micro-optimization. A list comprehension should fit on one line. If it spans three lines, use a regular `for` loop or break it into named helper functions. Code is read 10x more than it is written.

### Analogy
Think of a **photocopier with a built-in highlighter**:
- A `for` loop is like making a copy, walking to your desk, highlighting it, walking back, and stacking it. You do this 100 times.
- A list comprehension is the photocopier that *automatically highlights while copying* and stacks the finished pages in one continuous motion. Same result, less motion, fewer chances to drop a page.

---

## 2. Lambda Functions

### Topic
Anonymous, single-expression functions defined inline using the `lambda` keyword.

### Why It Is Related
AI pipelines are full of one-off transformations: "sort by confidence score," "filter out empty responses," "map labels to integers." Defining a full `def` function for each of these pollutes your namespace with throwaway names. `lambda` lets you define the logic exactly where you use it.

### How It Works
`lambda arguments: expression`
- Creates a function object without a name.
- Can be assigned to a variable, passed as an argument, or used inline.
- Restricted to a single expression (no statements, no assignments, no loops).

### Detailed Brief: How It Works & Valid Points
- **With `sorted()`**: `sorted(records, key=lambda r: r['score'])` sorts a list of dictionaries by a specific key. Without lambda, you would need a 3-line named function for a 1-second operation.
- **With `filter()`**: `filter(lambda x: x > 0, scores)` removes negative logits or invalid probabilities from model outputs.
- **With `map()`**: `map(lambda x: x.lower(), tokens)` normalizes text tokens. In practice, prefer list comprehensions for simple maps (they are clearer), but `map` with lambda is still common in functional programming pipelines.
- **In DataFrames**: `df.apply(lambda row: row['a'] + row['b'], axis=1)` is ubiquitous in Pandas preprocessing.

**Valid Point**: Lambda is a tool, not a religion. If the logic exceeds one line, use a named `def`. PEP 8 and every senior engineer will thank you. The goal is clarity, not brevity.

### Analogy
Think of lambda as a **disposable coffee filter**:
- You need it for one specific brew. You put the grounds in, pour water through, and toss it.
- You do not name it "CoffeeFilterNumberSeven," store it in a drawer, and maintain it for years.
- If you need a reusable gold filter (complex, multi-step logic), you buy a named `def` and wash it after each use.

---

## 3. Handling JSON/YAML for Model Configs

### Topic
Using Python to read, write, and manipulate JSON and YAML — the two dominant configuration formats in AI infrastructure.

### Why It Is Related
Modern AI systems are configured, not coded. A model's hyperparameters, a RAG pipeline's retrieval settings, an API's retry policy — all live in config files. Hardcoding these into Python means every tweak requires a git commit and code review. Config files let data scientists and engineers iterate without touching source code.

### How It Works
- **JSON (JavaScript Object Notation)**: A strict, machine-friendly format. Keys must be double-quoted. No comments allowed. Fast to parse. Used for schemas, API payloads, and inter-service communication.
- **YAML (YAML Ain't Markup Language)**: A human-friendly superset of JSON. Supports comments, multiline strings, anchors (`&`), and aliases (`*`). Used for Docker Compose, Kubernetes manifests, GitHub Actions, and ML experiment configs.
- **Python Integration**: `json.load()` / `json.dump()` for JSON. `yaml.safe_load()` / `yaml.dump()` for YAML. Both convert to native Python `dict` and `list` structures.

### Detailed Brief: How It Works & Valid Points
- **Why YAML for configs?** Because humans edit configs. A comment explaining `learning_rate: 0.001  # Reduced after epoch 5` prevents accidents. JSON cannot store that comment.
- **Why JSON for schemas?** Because schemas are contracts between machines. JSON Schema validators enforce type safety, required fields, and enum constraints automatically — critical when passing data between microservices.
- **Security Warning**: Never use `yaml.load()` (unsafe). It can execute arbitrary Python code embedded in YAML. Always use `yaml.safe_load()`. This is not theoretical — it is a CVE-class vulnerability.
- **Config Loading Pattern**: Load config once at startup, validate it against a schema, and pass the resulting dictionary to constructors. This is the "dependency injection" pattern for configuration.

**Valid Point**: In AI Engineering, the boundary between "code" and "config" is sacred. Code changes require tests and review. Config changes can be hot-swapped. Keeping them separate is how you move fast without breaking things.

### Analogy
Think of JSON and YAML as **blueprints vs. sticky notes**:
- **JSON** is the *architectural blueprint* handed to the construction crew. It is precise, standardized, and has no room for doodles. If the blueprint says "door is 36 inches," that is final.
- **YAML** is the *sticky notes on the blueprint* that say "Use the quieter hinge — tenant complained about noise" or "Paint this room blue, not beige." It is human-readable context that the blueprint format (JSON) cannot carry.
- Your Python code is the *construction crew*. It reads both and builds the house.

---

## 4. Error Handling for API Failures

### Topic
Writing resilient code that survives network failures, timeouts, rate limits, and server errors when calling external AI APIs.

### Why It Is Related
Your AI application is only as reliable as its weakest network call. OpenAI, Anthropic, Hugging Face, and internal model servers all fail. Rate limits (429), gateway timeouts (504), connection resets, and DNS blips are not exceptions — they are *expected operating conditions*. If your code crashes on the first retry, it is not production-grade.

### How It Works
1. **Exception Hierarchy**: Python's `requests` library raises specific exceptions:
   - `ConnectionError`: DNS failure, refused connection, network down.
   - `TimeoutError`: Server did not respond within the timeout window.
   - `HTTPError` (via `response.raise_for_status()`): 4xx/5xx status codes.
2. **Retry Logic**: Wrap the API call in a loop. On transient errors (network, timeout), wait and retry. On permanent errors (400 Bad Request), fail fast — retrying will never help.
3. **Exponential Backoff**: Increase wait time between retries (1s, 2s, 4s, 8s). This prevents your client from hammering a struggling server into the ground.
4. **Circuit Breaker** (advanced): After N consecutive failures, stop calling the API for a cooldown period. Prevents cascading failures in distributed systems.

### Detailed Brief: How It Works & Valid Points
- **Transient vs. Permanent**: 
  - Transient (retry): 500, 502, 503, 504, ConnectionError, TimeoutError.
  - Permanent (do not retry): 400, 401, 403, 404, 422.
- **Idempotency**: If your API call charges money or writes to a database, ensure it is idempotent (safe to retry). Use idempotency keys (UUIDs passed in headers) so the server recognizes duplicate requests.
- **Logging**: Every failure must be logged with context (timestamp, endpoint, payload hash, retry count). In production, you cannot debug what you cannot see.
- **Fallbacks**: If the primary LLM API fails after all retries, can you fall back to a local model (Ollama) or a cached response? This is how you build 99.9% uptime systems.

**Valid Point**: Most student projects ignore error handling because "the API usually works." In production, "usually" is a synonym for "unacceptable." A senior AI engineer is distinguished not by how they handle success, but by how they handle failure.

### Analogy
Think of API error handling as **driving in bad weather**:
- **No error handling** is driving 70mph on ice with no seatbelt. Most of the time, the road is straight and you are fine. One patch of black ice and you are totaled.
- **Basic try/except** is a seatbelt. It prevents you from flying through the windshield, but the car still crashes.
- **Retry with backoff** is anti-lock brakes + traction control. The car senses the skid, pumps the brakes automatically, and slows down progressively until grip returns.
- **Circuit breaker + fallback** is a GPS that reroutes you to a safer road when it detects the highway is closed, and a spare tire in the trunk just in case.

---

## Class 2 Achievement Map
| Achievement | Knowledge Applied |
|-------------|-------------------|
| Parse raw data into clean format | List comprehensions, lambda, dict transformations |
| Handle JSON/YAML configs | `json.load`, `yaml.safe_load`, schema validation |
| Survive API failures | `try/except`, retry loops, exponential backoff, transient vs. permanent error classification |

---

*End of Class 2 Lecture Material*
