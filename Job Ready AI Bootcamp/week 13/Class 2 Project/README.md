# ✈️ Travel Agent (Hand-written ReAct Loop)

## সহজ ভাষায় Project Overview

এই project-এ lecture-এর ReAct theory একটা কাজের agent-এ রূপ নিয়েছে। Agent local `data/flights.json` থেকে ফ্লাইট খোঁজে এবং একটা ফ্লাইট (simulated) বুক করে। Loop-টা framework ছাড়া নিজের হাতে লেখা (`src/react_agent.py`, প্রায় ১০০ লাইন), LLM চলে Ollama-তে।

**Flow:** Task → LLM (Thought + Action JSON) → tool চলে → Observation ফেরত → ... → Final Answer।

**চার লাগাম:** `max_steps`, LLM retry, tool/parse error-কে Observation বানানো, loop detection। সব সংখ্যা `config.py`-তে।

## Quick Start

```bash
# (ঐচ্ছিক) Ollama চালু ও model pull
ollama serve
ollama pull llama3.1:8b          # তুলনার জন্য: mistral:7b, llama3.2:3b

python app.py "Book the cheapest DAC to CXB flight on 2026-11-05 for Rahim"
python app.py --model llama3.2:3b "..."
python app.py --compare "..."    # তিন model পাশাপাশি
```

Ollama না থাকলে agent crash করে না: কয়েকবার retry করে `llm_failed` status আর সহায়ক message দেখায়। Runtime dependency শূন্য (stdlib only); `pytest` শুধু test-এর জন্য।

## Structure

```
config.py            frozen dataclass config + model reference table
app.py               CLI (--model, --compare)
data/flights.json    ১০টা ফ্লাইট (একটা sold out: BG102)
data/bookings.json   booking হলে তৈরি হয় (git-ignored)
src/tools.py         search_flights, book_flight, dispatch, ToolError
src/react_agent.py   run_agent(): পুরো ReAct loop
src/llm_client.py    Ollama /api/chat (stdlib urllib)
tests/test_agent.py  scripted fake LLM, Ollama ছাড়া
```

## Tests

```bash
pytest tests -q
```

Covered: search filter/sort, booking rules (sold out, duplicate, unknown id), happy path (search then book), tool error থেকে recovery, ভাঙা JSON ও unknown tool, `max_steps`, loop detection, LLM retry ও exhaustion।

## Model তুলনা

| Model | Use Case | Advantage | Limitation |
| --- | --- | --- | --- |
| `llama3.1:8b` (default) | Multi-step reasoning + tool use | JSON ও instruction ভালো মানে | ~16GB RAM |
| `mistral:7b` | দ্রুত agent | কম latency | Format মাঝে মাঝে ভাঙে |
| `llama3.2:3b` | Low-RAM মেশিন | ছোট ও দ্রুত | বেশি step-এ হারায়, loop-এ আটকায় |

## Known limits

* Booking শুধু simulated: `bookings.json`-এ লেখা হয়, flights.json-এর seat কমে না (শুধু memory-তে)। Concurrent লেখার lock নেই।
* Loop detection শুধু হুবহু একই action ধরে; ঘুরিয়ে করা repeat ধরে `max_steps`।
* Live Ollama-র সাথে তিন model-এর আচরণ এই repo-তে যাচাই করা হয়নি; tests fake LLM-এ।
