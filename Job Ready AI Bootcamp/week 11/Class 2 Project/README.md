# 🧾 Resume-to-JSON Converter

## সহজ ভাষায় Project Overview

এই notebook একটি unstructured CV text থেকে name, skill, experience এবং education বের করে **Pydantic-validated JSON** বানায়। Local LLM, LangChain prompt template এবং output parser একসাথে কীভাবে production-style information extraction করে সেটি দেখানো হয়েছে।

## কীভাবে কাজ করে?

```text
Resume Text → PromptTemplate → Local Ollama LLM
            → Pydantic Parser → Validated JSON
```

## Requirements

- Python ও Jupyter
- Local Ollama service
- `llama3.1:8b` model
- `langchain`, `langchain-ollama`, `pydantic`

```powershell
ollama pull llama3.1:8b
python -m pip install jupyter langchain langchain-ollama pydantic
cd "week 11\Class 2 Project"
python -m jupyter notebook resume_to_json.ipynb
```

## Validation Checklist

- Ollama service running
- Output valid JSON এবং Pydantic schema pass করে
- Missing field predictableভাবে handle হয়
- Prompt injection text schema bypass করতে পারে কি না test করা
- Sensitive CV data log বা external service-এ পাঠানো হয় না

> Notebook code-এর full execution-এর জন্য local Ollama এবং downloaded model mandatory।
