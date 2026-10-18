# Tool-Calling Agent: Math ও Sandboxed File Inspector

## Project Overview

এই project-এ একটা local LLM-কে "hands" দেওয়া হয়েছে: model নিজে অঙ্ক কষে না, file-ও খোলে না। সে শুধু সিদ্ধান্ত নেয় কোন tool কোন argument দিয়ে ডাকতে হবে, আর Python সেটা চালিয়ে result ফেরত দেয়। Model চলে Ollama-তে (`llama3.1:8b`), কোনো cloud API লাগে না।

## Architecture

```
question -> run_agent loop -> Ollama (tools schema সহ)
                 ^                 |
                 |          tool_calls? -- না --> final answer
                 |                 | হ্যাঁ
          tool result  <-- run_tool (calculate | list_files | read_file)
```

| Path | কাজ |
| --- | --- |
| `config.py` | model, host, sandbox path, max_steps; model comparison table |
| `src/tools/calculator.py` | AST-ভিত্তিক safe math, `eval()` ছাড়া |
| `src/tools/filesystem.py` | sandbox-এ বন্দি read-only file tools |
| `src/tools/registry.py` | JSON schema + dispatcher |
| `src/llm/client.py` | একমাত্র `ollama` import; failure -> `OllamaConnectionError` |
| `src/agent/loop.py` | agent loop, `max_steps` guard |
| `sandbox/` | demo ফাইল (`notes.txt`, `scores.csv`) |

## Quick Start

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b            # কম RAM হলে llama3.2:3b
python app.py "What is (12+8)*3 divided by 4?"
python app.py "What is the launch code in notes.txt?"
pytest tests -q                    # Ollama ছাড়াই চলে
```

Ollama বন্ধ থাকলে app crash করে না, একটা friendly message দেখায়।

## Safety

- Calculator: AST whitelist, বড় exponent বন্ধ।
- Files: `resolve()` + `is_relative_to` দিয়ে `..`, absolute path ও symlink escape আটকানো; শুধু read, কোনো write/delete নেই।
- Loop: `max_steps=5`; tool error exception না হয়ে string হিসেবে model-এর কাছে ফেরত যায়।

## Tests

`tests/test_agent.py` LLM-কে scripted fake function দিয়ে বদলে দেয়, তাই Ollama লাগে না। Cover করে: calculator injection, sandbox escape, unknown tool, agent loop, max_steps, connection failure।
