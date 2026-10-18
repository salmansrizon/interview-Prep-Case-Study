# Week 14 Class 2 Project — CrewAI Report Team

Researcher -> Writer -> Reviewer ক্রু, local `llama3.1:8b` (Ollama) চালিয়ে একটা topic-এর full report Markdown ফাইলে লেখে।

## Run

```bash
python -m venv <repo-এর বাইরে>/venv   # venv repo-তে রাখবেন না (Python 3.10-3.13)
pip install -r requirements.txt
ollama pull llama3.1:8b
python app.py "Vector databases" --mode sequential   # parallel | hierarchical
pytest -q                                            # Ollama/network লাগে না
```

Report যায় `output/` ফোল্ডারে। Ollama বন্ধ থাকলে app পরিষ্কার message দিয়ে exit করে।

## Structure

`config.py` (model + reference table) · `src/llm.py` · `src/agents.py` · `src/tasks.py` · `src/crew.py` (3 mode) · `src/report.py` · `app.py` · `tests/`

## Multi Agent Research Bot-এর সাথে সম্পর্ক

`Complete Agentic AI/Multi Agent Research Bot` হলো hand-rolled pipeline (Tavily search + Streamlit + cloud LLM)। এটা তার framework-ভিত্তিক শিক্ষামূলক রূপ: কোনো web search বা API key নেই, শুধু CrewAI-এর Agent/Task/Crew/Process concept, আর পুরোপুরি local।
