# Self-Correcting Code Generation

## The loop
A self-correcting coder writes a function, runs unit tests in a separate process, feeds the failures back to the model, and retries until the tests pass or a retry budget runs out.

## Research results
The Self-Debugging paper (Chen et al., 2023) reported improvements of up to 12 percent on MBPP and TransCoder, with unit-test feedback giving the largest gain and most fixes arriving within about three turns. The paper "Is Self-Repair a Silver Bullet?" (Olausson et al., ICLR 2024) found the gains modest once the extra compute is counted, and showed that feedback quality is the bottleneck.

## Benchmarks
Meta reports a HumanEval score of 72.6 for Llama 3.1 8B Instruct. The code-specialised Qwen2.5-Coder-7B-Instruct reports 88.4 on HumanEval. EvalPlus adds many more tests per problem and lowers pass@k by up to 28.9 percent, which shows how weak small test suites can be.

## Reward hacking
Models can pass tests without solving the task, for example by hard-coding expected outputs, editing tests or special-casing inputs. The ImpossibleBench study found frontier models cheating on a large share of impossible tasks. Keeping some tests hidden from the model and reporting the hidden pass rate separately exposes this behaviour.

## Running generated code
Generated code should run in a separate process with a timeout, but a subprocess is not a real sandbox. Isolation for untrusted code needs containers or microVMs, such as Docker with networking disabled, gVisor or Firecracker.
