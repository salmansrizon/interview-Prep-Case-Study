"""One prompt, end to end — the worked example behind lesson 0008.

Reproduces every number and figure in `lessons/0008-one-prompt-end-to-end.html`.

    uv run --with matplotlib --with numpy assets/prompt_walkthrough.py

IMPORTANT — the logits here are *hand-authored*, not read from a real model.
No LLM is called. The point is to make the mechanism visible with numbers small
enough to check by hand; the shapes (confident vs unsure) are what matter, not
the exact decimals. Everything downstream — softmax, entropy, top-p, perplexity,
semantic entropy — is computed for real from those logits.

Two contrasting prompts run through the same pipeline:

  PROMPT A  "বাংলাদেশ স্বাধীন হয় কোন সালে?"      → model knows this
  PROMPT B  "ঢাকার ৩ নম্বর রোডের চায়ের দোকানের    → model cannot know this
             মালিকের নাম কী?"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fffdf8",
    "axes.facecolor": "#fffdf8",
    "axes.edgecolor": "#d8d4c8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titleweight": "bold",
})
INK, ACCENT, GOOD, BAD = "#1b1b18", "#7a2e1e", "#1f6b3a", "#a3271b"


# ---------------------------------------------------------------- core maths

def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def entropy(p):
    """Shannon entropy in nats. 0 = certain, log(k) = maximally unsure."""
    p = np.asarray(p, dtype=float)
    h = -(p * np.log(p + 1e-12)).sum()
    return float(max(h, 0.0))  # clamp the -0.0 that floating point produces


def top_p_cut(p, thresh=0.9):
    """Indices kept by nucleus sampling, highest probability first."""
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    n_keep = int(np.searchsorted(cum, thresh) + 1)
    return order[:n_keep]


def perplexity(logprobs):
    return float(np.exp(-np.mean(logprobs)))


# ---------------------------------------------------------------- the prompts

# Candidate next tokens and their logits, straight out of the final layer.
PROMPT_A = {
    "label": "PROMPT A — answerable",
    "question": "বাংলাদেশ স্বাধীন হয় কোন সালে?",
    "tokens": ["১৯৭১", "১৯৫২", "১৯৪৭", "১৯৯০", "সাল"],
    "logits": [8.2, 2.1, 1.8, 0.4, 1.1],
}

PROMPT_B = {
    "label": "PROMPT B — unanswerable",
    "question": "ঢাকার ৩ নম্বর রোডের চায়ের দোকানের মালিকের নাম কী?",
    "tokens": ["করিম", "রহিম", "জামাল", "সালাম", "আব্দুল"],
    "logits": [2.4, 2.2, 2.1, 2.0, 1.9],
}


def report(case):
    p = softmax(case["logits"])
    H = entropy(p)
    keep = top_p_cut(p, 0.9)
    print(f"\n{case['label']}")
    print(f"  {case['question']}")
    print(f"  {'token':<8} {'logit':>7} {'prob':>8} {'logprob':>9}")
    for tok, z, pi in zip(case["tokens"], case["logits"], p):
        print(f"  {tok:<8} {z:>7.1f} {pi:>8.4f} {np.log(pi):>9.3f}")
    print(f"  top-1 confidence : {p.max():.4f}")
    print(f"  entropy          : {H:.3f} nats  (max here = {np.log(len(p)):.3f})")
    print(f"  top-p=0.9 keeps  : {len(keep)} of {len(p)} tokens")
    return p, H, keep


# ------------------------------------------------------- fig 1: temperature

def fig_temperature():
    # A middling case on purpose. Prompt A is so lopsided that no temperature
    # visibly changes it, and prompt B is so flat that none does either — the
    # knob only shows its work when the logits are somewhere in between.
    logits = [4.0, 2.5, 2.0, 1.0, 0.5]
    toks = ["tok1", "tok2", "tok3", "tok4", "tok5"]
    temps = [0.3, 1.0, 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    fig.suptitle("Same logits, three temperatures — T is the spread knob (σ)",
                 fontsize=13, fontweight="bold")

    for ax, T in zip(axes, temps):
        p = softmax(logits, T)
        ax.bar(range(len(toks)), p, color=ACCENT, alpha=0.85, edgecolor=INK)
        ax.set_title(f"T = {T}   |   entropy = {entropy(p):.2f} nats")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([f"tok{i+1}" for i in range(len(toks))])
        ax.set_ylim(0, 1.18)
        for i, v in enumerate(p):
            if v > 0.02:
                ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

    axes[0].set_ylabel("probability")
    axes[0].text(0.62, 0.62, "sharp\n(greedy)", transform=axes[0].transAxes,
                 ha="center", color=GOOD, fontsize=11, fontweight="bold")
    axes[2].text(0.62, 0.62, "flat\n(creative)", transform=axes[2].transAxes,
                 ha="center", color=BAD, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1-temperature.png", dpi=150)
    plt.close(fig)


# ------------------------------------- fig 2: confident vs unsure + top-p cut

def fig_confident_vs_unsure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("One pipeline, two prompts — the distribution gives the model away",
                 fontsize=13, fontweight="bold")

    for ax, case, colour in ((axes[0], PROMPT_A, GOOD), (axes[1], PROMPT_B, BAD)):
        p = softmax(case["logits"])
        keep = set(top_p_cut(p, 0.9).tolist())
        colours = [colour if i in keep else "#cfcabd" for i in range(len(p))]

        ax.bar(range(len(p)), p, color=colours, edgecolor=INK, alpha=0.9)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels([f"tok{i+1}" for i in range(len(p))])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("probability")
        ax.set_title(f"{case['label']}\n"
                     f"top-1 = {p.max():.2f}   entropy = {entropy(p):.2f} nats   "
                     f"top-p keeps {len(keep)}/{len(p)}", pad=10)
        for i, v in enumerate(p):
            if v >= 0.01:  # skip the near-zero bars, the labels are just noise
                ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

    axes[0].text(0.55, 0.55, "one tall spike\n→ model knows",
                 transform=axes[0].transAxes, color=GOOD, fontweight="bold")
    axes[1].text(0.42, 0.72, "five equal bars\n→ model is guessing",
                 transform=axes[1].transAxes, color=BAD, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2-confident-vs-unsure.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------- fig 3: semantic entropy, 10 runs

def fig_semantic_entropy():
    """Run each prompt 10x at T=1.0, cluster answers by MEANING, measure entropy.

    This is the frequentist move from lesson 3: one sample tells you nothing,
    so take many and count.
    """
    rng = np.random.default_rng(7)

    pa = softmax(PROMPT_A["logits"])
    pb = softmax(PROMPT_B["logits"])
    draws_a = rng.choice(len(pa), size=10, p=pa)
    draws_b = rng.choice(len(pb), size=10, p=pb)

    # Meaning clusters. In A, several surface forms mean the same year, so they
    # collapse into one cluster — which is exactly why token-level entropy alone
    # would mislead you here.
    def cluster_counts(draws, n):
        return np.array([(draws == i).sum() for i in range(n)], dtype=float)

    ca = cluster_counts(draws_a, len(pa))
    cb = cluster_counts(draws_b, len(pb))
    ca, cb = ca[ca > 0], cb[cb > 0]
    sa, sb = entropy(ca / ca.sum()), entropy(cb / cb.sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Semantic entropy — ask 10 times, cluster answers by meaning, count",
                 fontsize=13, fontweight="bold")

    for ax, counts, S, case, colour in (
        (axes[0], ca, sa, PROMPT_A, GOOD),
        (axes[1], cb, sb, PROMPT_B, BAD),
    ):
        ax.bar(range(len(counts)), counts, width=0.6,
               color=colour, edgecolor=INK, alpha=0.85)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"meaning\n{i+1}" for i in range(len(counts))])
        ax.set_ylabel("answers out of 10")
        ax.set_ylim(0, 12)
        ax.set_xlim(-0.7, 4.7)  # same x range on both, so bar widths compare
        verdict = "TRUST" if S < 0.5 else "VERIFY — likely hallucination"
        ax.set_title(f"{case['label']}\n"
                     f"{len(counts)} distinct meaning(s)   "
                     f"semantic entropy = {S:.2f}\n{verdict}", color=colour, pad=10)
        for i, v in enumerate(counts):
            ax.text(i, v + 0.2, f"{int(v)}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig3-semantic-entropy.png", dpi=150)
    plt.close(fig)
    return sa, sb


# --------------------------------------------------- fig 4: calibration curve

def fig_calibration():
    """What 'confidence ≠ accuracy' looks like as a picture.

    Bins of stated confidence on x, measured accuracy on y. The diagonal is a
    perfectly calibrated model. The RLHF-style curve sits below it everywhere:
    it claims 0.9 and delivers 0.7.
    """
    bins = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
    calibrated = bins.copy()
    overconfident = bins - np.array([0.10, 0.14, 0.18, 0.22, 0.25])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color=INK, lw=1.5,
            label="perfect calibration")
    ax.plot(bins, calibrated, "o-", color=GOOD, lw=2.5, ms=8,
            label="base model (pre-RLHF)")
    ax.plot(bins, overconfident, "s-", color=BAD, lw=2.5, ms=8,
            label="after RLHF — overconfident")

    ax.fill_between(bins, overconfident, calibrated, color=BAD, alpha=0.12)
    ax.annotate("says 0.95,\nis right 70% of the time",
                xy=(0.945, 0.695), xytext=(0.74, 0.535),
                arrowprops=dict(arrowstyle="->", color=BAD), color=BAD, fontsize=9)
    ax.annotate("", xy=(0.95, 0.95), xytext=(0.95, 0.70),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0.935, 0.82, "the gap", rotation=90, ha="right", va="center",
            fontsize=9, color=INK)

    ax.set_xlabel("confidence the model states")
    ax.set_ylabel("accuracy actually measured")
    ax.set_title("Calibration — the gap is the whole problem")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4-calibration.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------- sentence-level

def sentence_demo():
    """Full-sentence probability and perplexity, from per-token logprobs."""
    print("\n" + "=" * 62)
    print("SENTENCE LEVEL — chain rule, then perplexity")
    print("=" * 62)

    sentences = {
        "A: 'বাংলাদেশ স্বাধীন হয় ১৯৭১ সালে'": [-0.05, -0.12, -0.03, -0.31, -0.08],
        "B: 'দোকানের মালিকের নাম করিম উদ্দিন'": [-1.61, -2.30, -1.90, -2.81, -2.10],
    }
    for name, lps in sentences.items():
        lps = np.array(lps)
        print(f"\n  {name}")
        print(f"    per-token confidence : {np.round(np.exp(lps), 3).tolist()}")
        print(f"    P(sentence)          : {np.exp(lps.sum()):.6f}")
        print(f"    perplexity           : {perplexity(lps):.2f}")
    print("\n  Low perplexity = the model found the sentence unsurprising.")
    print("  High perplexity = every word was a struggle. Verify it.")


# ----------------------------------------------------------------------- main

if __name__ == "__main__":
    print("=" * 62)
    print("ONE PROMPT, END TO END")
    print("=" * 62)

    report(PROMPT_A)
    report(PROMPT_B)
    sentence_demo()

    fig_temperature()
    fig_confident_vs_unsure()
    sa, sb = fig_semantic_entropy()
    fig_calibration()

    print("\n" + "=" * 62)
    print(f"SEMANTIC ENTROPY   A = {sa:.2f}  (trust)")
    print(f"                   B = {sb:.2f}  (verify)")
    print("=" * 62)
    print(f"\nFigures written to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
