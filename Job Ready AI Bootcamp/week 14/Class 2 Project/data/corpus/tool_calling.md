# Tool Calling with Local Models

## How it works
In tool calling the model never runs code. It returns a structured request containing a tool name and JSON arguments. The host program validates and runs the tool, sends the result back as a tool message, and calls the model again until it answers without a tool call.

## Ollama specifics
In Ollama the tool arguments arrive as an already-parsed dictionary, while OpenAI returns them as a JSON string. Ollama matches tool results to calls by their order, without a documented tool_call_id. Ollama's default temperature is 0.8, so agents should set temperature explicitly. The OpenAI-compatible endpoint of Ollama does not support tool_choice.

## Model reliability
On the Berkeley Function Calling Leaderboard (BFCL v4, April 2026), Llama-3.1-8B-Instruct ranks 85th with 25.83 percent overall, 11.12 percent on multi-turn tasks and 42.70 percent on irrelevance detection. Qwen3-8B scores 42.57 percent overall and 41.75 percent on multi-turn. Low irrelevance detection means a model calls tools when no tool fits.

## Safe tools
A calculator tool must never use eval. LangChain's LLMMathChain vulnerability (CVE-2023-29374) allowed arbitrary code execution through exactly this mistake. A safe calculator parses the expression with the ast module and evaluates only an allowlist of numbers and operators. A file tool should resolve the path first and then check that it stays inside a sandbox directory.
