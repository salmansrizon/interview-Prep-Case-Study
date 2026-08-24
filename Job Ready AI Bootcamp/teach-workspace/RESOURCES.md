# Resources

High-trust sources used to ground the lessons. All URLs verified reachable on 2026-08-12.

## Primary — visual & interactive

| Resource | Covers | Why trusted | Link |
|---|---|---|---|
| Seeing Theory (Brown University) | Distributions, basic probability, Bayesian inference | University-published, interactive simulations you can drive yourself | https://seeing-theory.brown.edu/basic-probability/index.html |
| Seeing Theory — Bayesian Inference | Prior, likelihood, posterior, Beta distribution | Same; the Beta section directly backs lesson 5 | https://seeing-theory.brown.edu/bayesian-inference/index.html |
| 3Blue1Brown — Bayes' theorem | Bayes, the medical-test paradox | Best visual explanation available; uses the same "count 10,000 people" framing as lesson 4 | https://www.3blue1brown.com/lessons/bayes-theorem |
| 3Blue1Brown — Binomial distributions / Bayes video | Frequentist intuition | Video form | https://www.youtube.com/watch?v=HZGCoVF3YvM |

## Practice

| Resource | Covers | Link |
|---|---|---|
| Khan Academy — Modeling Distributions of Data | Z-scores, percentiles, empirical rule; graded exercises | https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data |
| StatQuest video index | Short, plain-language stats videos for every topic here | https://statquest.org/video-index/ |

## Documentation — the authority for the code in the notebooks

| Resource | Covers | Link |
|---|---|---|
| scipy.stats.norm | `pdf`, `cdf`, `ppf` used throughout the notebook | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html |
| scikit-learn — Probability calibration | Calibration curves, Platt scaling, isotonic regression | https://scikit-learn.org/stable/modules/calibration.html |

## Communities — where wisdom comes from

Not yet chosen. Worth proposing once the mission is confirmed:

- r/learnmachinelearning and r/statistics — good for "is my interpretation of this CI right?"
- Cross Validated (stats.stackexchange.com) — the highest-signal place for statistics
  questions specifically; answers are heavily peer-corrected.
- Local: Bangladesh-based ML/data meetup groups, for Bangla-language practice.

**Ask the user first** — no community has been recommended to them yet, and they may
not want one.

## LLM connection (lessons 6–7)

All titles fetched and verified 2026-08-12.

| Resource | Covers | Link |
|---|---|---|
| Ba, Kiros & Hinton — *Layer Normalization* | LayerNorm = z-score, eq. 2 | https://arxiv.org/abs/1607.06450 |
| Zhang & Sennrich — *Root Mean Square Layer Normalization* | RMSNorm, used by Llama/T5/Gemma | https://arxiv.org/abs/1910.07467 |
| Vaswani et al. — *Attention Is All You Need* | §3.2.1 explains the `/√d_k` variance argument | https://arxiv.org/abs/1706.03762 |
| Holtzman et al. — *The Curious Case of Neural Text Degeneration* | Top-p / nucleus sampling, why pure temperature fails | https://arxiv.org/abs/1904.09751 |
| Kadavath et al. — *Language Models (Mostly) Know What They Know* | LLM calibration, P(True) self-evaluation | https://arxiv.org/abs/2207.05221 |
| Kuhn, Gal & Farquhar — *Semantic Uncertainty* | Meaning-clustered entropy for hallucination detection | https://arxiv.org/abs/2302.09664 |
| Farquhar et al., *Nature* (2024) | Peer-reviewed follow-up on semantic entropy | https://www.nature.com/articles/s41586-024-07421-0 |
| Guo et al. — *On Calibration of Modern Neural Networks* | Modern nets are overconfident; temperature scaling as the fix | https://arxiv.org/abs/1706.04599 |
| Hugging Face — Perplexity | Perplexity with runnable code | https://huggingface.co/docs/transformers/perplexity |
| GPT-4 Technical Report | Calibration before vs after RLHF | https://cdn.openai.com/papers/gpt-4.pdf |

Note: the OpenAI API reference for `logprobs` blocks automated fetching (403). Reachable
in a browser; not linked from lessons for that reason.

## Bangla-language sources

None found yet at the required quality bar. Gap worth filling: if the mission turns out
to be *teaching* Bangla-speaking students, a survey of Bangla stats resources becomes
high-value. Until then, English sources with Bangla lessons on top.
