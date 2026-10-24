# Running Local Models with Ollama

## Why local
A local model downloads its weights once and then runs entirely on the user's machine, so prompts and documents never leave it. This course uses Ollama with the llama3.1:8b model as its standard local model.

## Defaults to change
Ollama's default sampling temperature is 0.8, which is too random for agents and tool use. Its documentation gives the default context window as 4096 tokens in the FAQ, while the Modelfile reference lists 2048, so applications should set num_ctx explicitly. By default Ollama processes one request at a time per model, so parallel agent tasks on one Ollama server mostly wait in a queue.

## Model size and memory
An 8-billion-parameter model quantized to 4 bits needs roughly 5 gigabytes of memory. Smaller models are faster but weaker; on BFCL v4 the 3-billion-parameter Llama 3.2 scores only 4 percent on multi-turn function calling.

## The Python client
The ollama Python library's chat function accepts messages, tools, a format for structured output, and an options dictionary for settings such as temperature, seed and num_ctx. Since version 0.4 it returns typed response objects, so code can read response.message.content and response.message.tool_calls.
