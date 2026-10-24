# ReAct Agents

## The paper
ReAct (Yao et al., 2022) prompts a model to interleave reasoning traces (Thought) with actions (Action) and to read the results (Observation). With PaLM-540B, ReAct scored 27.4 exact match on HotpotQA, slightly below chain-of-thought at 29.4, but reached 71 percent success on ALFWorld against 45 percent for an action-only baseline.

## Failure modes
In the paper's error analysis none of ReAct's HotpotQA failures were hallucinations, compared with 56 percent of chain-of-thought failures. Instead, 47 percent of ReAct failures were reasoning errors, including repetitive loops, and 23 percent came from uninformative search results.

## Engineering the loop
Because of these failure modes an agent loop needs hard stop conditions: a maximum number of steps, detection of repeated identical tool calls, a deadline, and a budget for consecutive invalid calls. Each stop should produce a typed reason so tests can check why the loop ended.

## Determinism
Temperature 0 does not guarantee identical outputs. Thinking Machines found 80 unique completions among 1,000 temperature-0 requests to the same model, because results depend on how requests are batched. Agent tools can be fully deterministic, but the language model cannot, so unit tests should mock the model.
