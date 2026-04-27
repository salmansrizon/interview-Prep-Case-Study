# Week 1 — Class 2 Project: Advanced Python for AI Data Parsing

## Achievement
Build a local Python script that parses raw data files into a clean format using:
- List comprehensions
- Lambda functions  
- JSON/YAML configuration handling
- Production-grade error handling for API failures

---

## Production-Grade File Structure

```
week1_class2_project/
├── README.md                     # You are here
├── requirements.txt              # Pinned dependencies
├── config/
│   ├── model_config.yaml         # Human-readable parser + API settings
│   └── schema.json               # JSON Schema for data validation
├── src/
│   ├── __init__.py               # Package marker
│   ├── config_loader.py          # YAML/JSON loader with caching
│   ├── cleaners.py               # List comprehensions, lambda, filter, map
│   ├── validators.py             # Schema validation (jsonschema)
│   ├── api_client.py             # Simulated API with retry logic
│   └── parser.py                 # Pipeline orchestrator
├── data/
│   ├── raw/                      # Input JSONL files
│   └── clean/                    # Output clean + quarantine files
└── main.py                       # Single entry point
```

---

## Step-by-Step Instructions

### Step 1: Set Up Virtual Environment

```bash
cd week1_class2_project
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

**ANALOGY:** This is laying a clean foundation before building a house. You do not pour concrete on a muddy field.

---

### Step 2: Inspect the Configuration Files

**`config/model_config.yaml`**
- Defines input/output paths, cleaning rules, and API retry settings.
- **WHY YAML?** Because it supports comments. Students can read WHY each setting exists.

**`config/schema.json`**
- Defines the exact shape of a valid record (required fields, types, enums).
- **WHY JSON Schema?** It is a contract. If a record fails validation, you know exactly which rule broke.

---

### Step 3: Run the Parser

```bash
python main.py
```

**What happens:**
1. `main.py` creates intentionally messy sample data in `data/raw/sample_raw.jsonl`.
2. `DataParser.run()` executes a 5-stage pipeline:
   - **Read**: Loads all `.jsonl` files from `data/raw/`.
   - **Clean**: Applies list comprehensions and lambda filters (strip whitespace, remove empty, filter status/length, sort).
   - **Validate**: Checks every record against `schema.json` using `jsonschema`.
   - **Enrich**: Calls the simulated API with retry logic (ConnectionError, Timeout, HTTP 500).
   - **Write**: Saves valid records to `clean_data.jsonl` and invalid ones to `quarantine_data.jsonl`.

---

### Step 4: Inspect the Output

```bash
cat data/clean/clean_data.jsonl
cat data/clean/quarantine_data.jsonl
```

**WHY two files?** In production, you never discard bad data. You quarantine it for forensic analysis.

---

### Step 5: Experiment with Config Changes

Edit `config/model_config.yaml`:

```yaml
cleaning:
  min_text_length: 20      # Reject very short records
  allowed_statuses:
    - "active"             # Only keep active records
```

Run again:
```bash
python main.py
```

**WHY?** To prove that behavior changes without touching Python code. This is *configuration-driven engineering*.

---

## Key Code Patterns Explained

### List Comprehensions (`src/cleaners.py`)
```python
[key: value.strip() if isinstance(value, str) else value 
 for key, value in record.items()]
```
- **WHY?** Faster and more readable than nested for-loops.
- **ANALOGY:** A conveyor belt with a built-in tool. Instead of grabbing each item, walking to the workbench, modifying it, and walking back, the modification happens as the item moves.

### Lambda + Filter (`src/cleaners.py`)
```python
filter(lambda r: r.get("text") and str(r.get("text")).strip(), records)
```
- **WHY?** Defines a throwaway predicate without polluting the namespace.
- **ANALOGY:** A disposable coffee filter. You use it once, it catches the grounds, and you toss it. You do not keep it in your kitchen drawer.

### Retry Logic (`src/api_client.py`)
```python
for attempt in range(1, max_retries + 1):
    try:
        return self.enrich_record(record)
    except (ConnectionError, TimeoutError):
        time.sleep(delay)
        delay *= 2  # Exponential backoff
```
- **WHY?** Networks are unreliable. Retry with backoff prevents hammering a struggling server.
- **ANALOGY:** Knocking on a friend's door. First knock: normal. Second: wait longer. Third: maybe they are not home. You do not bang continuously.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: jsonschema` | Run `pip install -r requirements.txt` |
| `No raw data found` | Run `main.py` — it auto-generates sample data |
| `All records quarantined` | Check `schema.json` required fields match your raw data |
| `API always fails` | This is intentional! The simulator fails every 5th/7th/9th call to teach retry logic. |

---

## Learning Checklist
- [ ] I can write a list comprehension that filters AND transforms data.
- [ ] I can explain when to use `lambda` vs. a named `def` function.
- [ ] I can load YAML and JSON configs safely in Python.
- [ ] I can implement retry logic with exponential backoff.
- [ ] I understand why schema validation prevents production outages.
- [ ] I can trace data flow: raw → clean → validate → enrich → output.
