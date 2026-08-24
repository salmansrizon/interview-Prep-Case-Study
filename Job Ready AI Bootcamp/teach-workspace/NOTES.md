# Notes

Working notes and user preferences for this teaching workspace.

## Stated preferences

- **Bangla, simple.** "Explain like a 15 year old, in simple Bangla." Short sentences,
  everyday analogies (class heights, exam marks, coin toss), no academic register.
- **Keep technical terms in English.** Z-score, prior, posterior, calibration, standard
  error. Translating these would make the notebooks and the interview room harder, not
  easier. Bangla carries the *explanation*; English carries the *nouns*.

## Workspace decisions

- Lessons live in `teach-workspace/` rather than the bootcamp root, so course content
  and teaching-workspace state stay separate in git.
- Every lesson names the notebook section it maps to (`Section B`, `Section D`, …) so
  lesson and code do not drift apart.
- Quiz options are written to near-equal length in Bangla, so option length gives no
  clue to the answer.

## Content gotchas found in the source notebook

- Cell 23 is empty.
- Several markdown cells are truncated mid-sentence (Section D "❌ Can't update…",
  Section E "False Positive…", Section G "Confidence ≠ Accuracy | Mus…"). The lessons
  fill these gaps; worth fixing in the notebook itself.
- The notebook computes the medical-test posterior but the surrounding prose stops
  short of the "count 10,000 people" intuition, which is the part that actually lands.
  Lesson 4 leads with it.

## Confirmed preference (session 1, second request)

- **Always land the LLM connection.** The follow-up to lessons 1–5 was "also add how
  it's implemented & used in LLM models." Treat this as standing guidance: for any
  statistics or maths topic in this bootcamp, the lesson is not finished until it says
  where the idea actually sits inside a model or an LLM workflow. Abstract stats does
  not motivate this audience; `LayerNorm is literally the z-score` does.

## Confirmed preference (session 1, third request)

- **Worked examples and pictures, not prose.** "Explain all this with a prompt example,
  visualize if needed." Concepts explained one-by-one didn't land the way one input
  traced end-to-end does. Default for future lessons: pick a single concrete artefact,
  run every concept over it in numbered steps, and draw the result.
- Prefer a **contrasting pair** over a single example. One case where it works and one
  where it fails makes every metric produce a comparison rather than a bare number.

## Environment

- matplotlib/numpy are **not** installed in the repo Python. Use
  `uv run --with matplotlib --with numpy <script>` — do not add them to the project
  environment without asking.

## Ideas for future lessons

- Calibration curve, hands-on, on one of the user's own models (offered at the end of
  lesson 5).
- Interleaved review lesson mixing all five topics — spaced a week out, for storage
  strength rather than fluency.
- Confusion matrix / precision / recall — the notebook imports them but never explains
  them.
- Hands-on notebook offered at the end of lesson 7: pull logprobs from a small open
  model, run one prompt ten times, compute semantic entropy. Highest-value practical
  follow-up available right now.
- Hands-on offered at the end of lesson 6: write LayerNorm in NumPy, diff it against
  `torch.nn.LayerNorm`.
