"""Central settings for the CrewAI report team (Week 14, Class 2)."""
import os
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("CREW_MODEL", "llama3.1:8b")
TEMPERATURE = 0.2
OUTPUT_DIR = Path(__file__).parent / "output"

# HIGHLIGHTS: কেন llama3.1:8b? Crew-এ প্রতিটা agent বারবার instruction follow করে আর
# "Thought/Action/Final Answer" format ধরে রাখতে হয় — তাই instruction-following ভালো model দরকার।
# 8B মাঝামাঝি: laptop-এ চলে, আবার ছোট model-এর মতো format ভাঙে না।
# Reference table (model -> কখন ব্যবহার করবেন / trade-off):
MODEL_TABLE = {
    "llama3.1:8b": "Default। ভালো instruction following, ~8GB RAM। Crew-এর জন্য balanced।",
    "mistral:7b": "দ্রুত, লেখার মান ভালো; format ধরে রাখতে মাঝে মাঝে ভুল করে।",
    "llama3.2:3b": "খুব হালকা (~4GB), দ্রুত; ছোট বলে reviewer/manager role-এ দুর্বল।",
    "qwen2.5:7b": "Structured output ও multilingual-এ শক্ত; ধীর হতে পারে।",
    "llama3.1:70b": "সেরা মান, কিন্তু ~40GB+ memory লাগে — hierarchical manager-এর জন্য আদর্শ।",
}
