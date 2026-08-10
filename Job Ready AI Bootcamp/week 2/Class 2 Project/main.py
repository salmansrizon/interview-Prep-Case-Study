"""
main.py
-------
Motive: Single entry point for the Semantic Similarity Engine.
WHAT IT DOES: Loads a corpus, embeds it into an ndarray, proves the memory
and broadcasting behaviour, benchmarks vectorization against a
Python loop, runs cosine search and attention, then grades the
student's exercises.
ANALOGY: The assembly line. Raw text goes in one end; a searchable vector
         index, a JSON report, and a scorecard come out the other.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

# WHY THIS BLOCK? So `python main.py` works no matter which directory you
# launch it from. PROJECT_ROOT is the folder holding this file; every path
# below is built from it instead of from your shell's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# These imports MUST come after the sys.path line above — that is what lets
# `python main.py` work from any directory. Linters flag it; moving them up
# breaks the script. pylint: disable=wrong-import-position
from src.attention import run_attention_demo
from src.embedding_store import EmbeddingStore
from src.similarity_engine import (
    SimilarityEngine,
    angle_between,
    cosine_similarity,
    dense_layer,
)
from src.skill_check import print_scorecard, run_skill_check
from src.utils.array_io import read_corpus, save_array, write_json_report
from src.vector_ops import (
    add_bias,
    benchmark_loop_vs_vectorized,
    benchmark_memory_layout,
    l2_normalize,
    standardize,
)

QUERIES = [
    "gpu memory model",
    "cats and dogs",
    "fast numpy code without loops",
]


def main() -> int:
    """
    Runs the whole pipeline top to bottom and returns a process exit code.

    WHY one long function instead of six small ones? This file is meant to be
    READ straight through, like the lecture it follows. Every part builds on
    the variables of the part above it, in the order the class taught them.
    The reusable logic already lives in src/ — this is the narration.
    """
    # pylint: disable=too-many-locals,too-many-statements
    print("=" * 70)
    print("SEMANTIC SIMILARITY ENGINE")
    print("Week 2, Class 4 — Numerical Processing with NumPy")
    print("=" * 70)

    # WHY ["engine"]? Every setting lives under the top-level `engine:` key.
    # Drop it and each module silently falls back to its defaults — no crash,
    # no warning, just a config file that quietly does nothing.
    with open(PROJECT_ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["engine"]

    paths = config.get("paths", {})

    # ============================================================
    # PART 1: BUILD THE EMBEDDING MATRIX (ndarray)
    # ============================================================
    print("\n[PART 1] Embedding the corpus into an ndarray...")
    documents = read_corpus(PROJECT_ROOT / paths.get("corpus_file", "data/raw/corpus.txt"))
    print(f"         Loaded {len(documents)} documents")

    store = EmbeddingStore(config)
    matrix = store.fit_transform(documents)

    info = store.describe()
    print("\n         Array structure:")
    print(f"         - shape       : {tuple(info['shape'])}  (documents × dimensions)")
    print(f"         - dtype       : {info['dtype']} ({info['itemsize_bytes']} bytes/element)")
    print(f"         - strides     : {tuple(info['strides'])}  bytes to jump per axis")
    print(f"         - C-contiguous: {info['c_contiguous']}")
    print(f"         - memory      : {info['memory_kb']} KB")
    print(f"         - sparsity    : {info['sparsity_pct']}% of the cells are zero")

    mem = store.memory_comparison()
    # Mark whichever dtype the config actually chose, instead of assuming float32.
    here = mem["current_dtype"]
    print("\n         Same numbers, different containers:")
    print(f"         - float32{' (current)' if here == 'float32' else '          '}: "
        f"{mem['as_float32_kb']:>8.2f} KB")
    print(f"         - float64{' (current)' if here == 'float64' else '          '}: "
        f"{mem['as_float64_kb']:>8.2f} KB   (2x)")
    print(f"         - Python list      : {mem['as_python_list_kb']:>8.2f} KB   (~8x)")

    save_array(matrix, PROJECT_ROOT / paths.get("vector_dir", "data/vectors") / "embeddings.npy")
    print("\n         ✓ Vectors saved to data/vectors/embeddings.npy")

    # ============================================================
    # PART 2: VIEWS, COPIES AND BROADCASTING
    # ============================================================
    print("\n[PART 2] Views vs. copies, and broadcasting...")
    vc = store.view_vs_copy_demo()
    print(f"         - matrix[0:2]        is a view? {vc['slice_is_view']}"
        "   (basic slicing shares memory)")
    print(f"         - matrix[0:2].copy() is a view? {vc['copy_is_view']}  (copy breaks the link)")
    print(f"         - matrix[[0,1]]      is a view? {vc['fancy_index_is_view']}"
        "  (fancy indexing always copies)")
    print("         - writing into the view changed the source: "
        f"{vc['source_before']} -> {vc['source_after_view_write']}")
    print(f"         - writing into the copy left it at: {vc['source_after_copy_write']}")

    bias = np.arange(matrix.shape[1], dtype=matrix.dtype)
    biased = add_bias(matrix, bias)
    print(f"\n         Broadcasting: {matrix.shape} + {bias.shape} -> {biased.shape}")
    print("         No copy of the bias is made — NumPy walks that axis with stride 0.")

    standardized = standardize(matrix, axis=0)
    # Columns that never fired are constant, so they have no spread to scale.
    # The EPSILON guard leaves them at 0 instead of dividing by zero -> NaN.
    live = matrix.std(axis=0) > 0
    print(f"         Standardized per feature: mean≈{standardized[:, live].mean():.6f}, "
        f"std≈{standardized[:, live].std():.4f}  "
        f"({int(live.sum())}/{live.size} features had any variance)")

    # ============================================================
    # PART 3: VECTORIZATION BENCHMARK
    # ============================================================
    bench_cfg = config.get("benchmark", {})
    if bench_cfg.get("enabled", True):
        print("\n[PART 3] Vectorization vs. Python loops...")
        bench = benchmark_loop_vs_vectorized(int(bench_cfg.get("n_elements", 1_000_000)))
        print(f"         sqrt(a² + b²) over {bench['n_elements']:,} elements:")
        print(f"         - Python loop     : {bench['python_loop_ms']:>8.2f} ms")
        print(f"         - NumPy vectorized: {bench['numpy_vectorized_ms']:>8.2f} ms")
        print(f"         - speedup         : {bench['speedup_x']}x")
        print(f"         - identical results: {bench['results_match']}")

        layout = benchmark_memory_layout(3000)
        print("\n         Memory layout (same numbers, same reduction):")
        print(f"         - C-order sum(axis=1): {layout['sum_axis1_c_order_ms']:>6.2f} ms   "
            f"strides {tuple(layout['c_order_strides'])}")
        print(f"         - F-order sum(axis=1): {layout['sum_axis1_f_order_ms']:>6.2f} ms   "
            f"strides {tuple(layout['f_order_strides'])}")
        print(f"         - transpose copies data: {layout['transpose_copies_data']}"
            "  (it only swaps strides)")
    else:
        bench, layout = {}, {}

    # ============================================================
    # PART 4: DOT PRODUCT GEOMETRY
    # ============================================================
    print("\n[PART 4] Dot product geometry...")
    cases = {
        "same direction": (np.array([1.0, 0.0]), np.array([5.0, 0.0])),
        "45 degrees    ": (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        "orthogonal    ": (np.array([1.0, 0.0]), np.array([0.0, 5.0])),
        "opposite      ": (np.array([1.0, 0.0]), np.array([-5.0, 0.0])),
    }
    print(f"         {'case':<16}{'dot':>10}{'cosine':>10}{'angle':>10}")
    for label, (u, v) in cases.items():
        print(f"         {label:<16}{u @ v:>10.2f}{cosine_similarity(u, v):>10.2f}"
            f"{angle_between(u, v):>9.1f}°")
    print("\n         Magnitude pollutes a raw dot product; cosine divides it out.")
    print("         That is why search normalizes — otherwise the longest document wins.")

    rng = np.random.default_rng(0)
    # W is capitalised on purpose: it is the standard symbol for a weight
    # matrix in every paper and framework. Renaming it to `w` would make the
    # code match a linter and stop matching the maths.
    x = rng.standard_normal((4, 6))
    W = rng.standard_normal((6, 3))  # pylint: disable=invalid-name
    b = rng.standard_normal(3)
    h = dense_layer(x, W, b)
    print("\n         One neural layer is just this: relu(x @ W + b)")
    print(f"         x{x.shape} @ W{W.shape} + b{b.shape} -> h{h.shape}")

    # ============================================================
    # PART 5: SEMANTIC SEARCH (the capstone)
    # ============================================================
    print("\n[PART 5] Cosine similarity search...")
    engine = SimilarityEngine(config).index(matrix, documents)
    sim_report = engine.report()

    print(f"         Index: {sim_report['n_documents']} docs × {sim_report['dimensions']} dims")
    print(f"         - diagonal is all 1.0 : {sim_report['diagonal_is_one']}")
    print(f"         - matrix is symmetric : {sim_report['matrix_is_symmetric']}")
    print(f"         - mean pair similarity: {sim_report['mean_off_diagonal_similarity']}")

    pair = sim_report["most_similar_pair"]
    if pair["score"] is None:
        print("\n         Most similar pair: n/a — a corpus of one has no pairs.")
    else:
        print(f"\n         Most similar pair (score {pair['score']}):")
        print(f"           A: {pair['doc_a']}")
        print(f"           B: {pair['doc_b']}")

    search_results = {}
    for query in QUERIES:
        q_vec = store.embed_one(query)
        hits = engine.search(q_vec)
        search_results[query] = [{"document": d, "score": round(s, 4)} for d, s, _ in hits]
        print(f"\n         query: {query!r}")
        for doc, score, _ in hits:
            print(f"           {score:.3f}  {doc}")

    if bench_cfg.get("enabled", True):
        n_docs = int(bench_cfg.get("n_docs", 20000))
        dim = int(bench_cfg.get("dimensions", 256))
        big = rng.standard_normal((n_docs, dim)).astype(np.float32)
        big = l2_normalize(big)
        q = big[0]
        t0 = time.perf_counter()
        loop_scores = np.array([float(np.dot(big[i], q)) for i in range(n_docs)])
        t_loop = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        vec_scores = big @ q
        t_vec = (time.perf_counter() - t0) * 1000
        print(f"\n         Searching {n_docs:,} docs × {dim} dims:")
        print(f"         - loop of np.dot: {t_loop:>7.2f} ms")
        speedup = t_loop / max(t_vec, 1e-9)
        print(f"         - single matmul : {t_vec:>7.2f} ms   ({speedup:.0f}x faster)")
        print(f"         - same answers  : {np.allclose(loop_scores, vec_scores, atol=1e-5)}")

    # ============================================================
    # PART 6: ATTENTION
    # ============================================================
    print("\n[PART 6] Scaled dot-product attention...")
    att = run_attention_demo(config)
    print(f"         Q@K.T -> scores {tuple(att['scores_shape'])}, "
        f"output {tuple(att['output_shape'])}")
    print(f"         - every row sums to 1 : {att['rows_sum_to_one']}")
    print(f"         - causal mask correct : {att['causal_structure_correct']}  "
        f"(non-zeros per row: {att['nonzero_per_row']})")
    print(f"         - max weight WITH  sqrt(d_k): {att['max_weight_scaled']}")
    print(f"         - max weight WITHOUT       : {att['max_weight_unscaled']}"
        "  <- softmax saturating")
    print("\n         Attention weights (rows = tokens, upper triangle blocked):")
    for row in att["weights"]:
        print("           " + "  ".join(f"{v:.3f}" for v in row))

    # ============================================================
    # SAVE THE REPORT
    # ============================================================
    report = {
        "array": info,
        "memory": mem,
        "views_and_copies": vc,
        "benchmark": bench,
        "memory_layout": layout,
        "similarity": sim_report,
        "searches": search_results,
        "attention": {k: v for k, v in att.items() if k != "weights"},
    }
    report_path = PROJECT_ROOT / paths.get("report_dir", "data/reports") / "similarity_report.json"
    write_json_report(report, report_path)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"✓ Documents indexed : {len(documents)}")
    print(f"✓ Embedding matrix  : {tuple(info['shape'])} {info['dtype']} "
          f"({info['memory_kb']} KB)")
    print(f"✓ Queries answered  : {len(QUERIES)}")
    print("✓ Vectors saved to  : data/vectors/embeddings.npy")
    print("✓ Report saved to   : data/reports/similarity_report.json")
    print("=" * 70)

    # ============================================================
    # SKILL CHECK — grades src/exercises.py
    # ============================================================
    sc_cfg = config.get("skill_check", {})
    if sc_cfg.get("enabled", True):
        card = run_skill_check()
        print_scorecard(card)
        report["skill_check"] = {
            "score": card["score"],
            "total": card["total"],
            "pct": card["pct"],
            "results": [{k: r[k] for k in ("name", "status", "detail")} for r in card["results"]],
        }
        write_json_report(report, report_path)

        if sc_cfg.get("strict", False) and not card["all_passed"]:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
