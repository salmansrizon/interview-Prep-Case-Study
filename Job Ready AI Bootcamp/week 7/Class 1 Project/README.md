# 🧠 Neural Network Lab

## সহজ ভাষায় Project Overview

**🧠 Neural Network Lab** project-এ lecture-এর theory-কে working software বা executable notebook-এ convert করা হয়েছে। লক্ষ্য শুধু final output দেখা নয়; input থেকে preprocessing, core logic/model, evaluation এবং output—পুরো pipeline বোঝা।

### কোন Problem Solve করে?

Manual বা disconnected workflow-কে repeatable code pipeline-এ আনে। এর ফলে একই process নতুন data-তে আবার চালানো, result compare করা, error trace করা এবং future feature add করা সহজ হয়।

### কীভাবে কাজ করে?

Input/Data → Validation ও Preprocessing → Core Algorithm/Model → Evaluation → UI, Report বা Saved Output। নিচের detailed section-গুলোতে project-specific command, feature এবং architecture দেওয়া আছে।

### কেন এই Approach ভালো?

- **Repeatable:** একই input দিলে একই workflow follow করে।
- **Testable:** প্রতিটি stage আলাদাভাবে verify করা যায়।
- **Explainable:** কোন step কী কাজ করছে তা code এবং output দিয়ে দেখা যায়।
- **Portfolio-ready:** শুধু notebook result নয়, setup, structure এবং usage-সহ complete project হিসেবে দেখানো যায়।

> **Run করার নিয়ম:** আগে virtual environment তৈরি করে dependency install করুন। তারপর README-এর Quick Start follow করুন, sample input দিয়ে smoke test করুন এবং expected metric/output-এর সাথে result compare করুন।

---

A production-grade, 100% offline educational platform for learning Neural Networks from scratch.

## Topics Covered
1. **The Neuron** — Weights, biases, and the perceptron model
2. **Activation Functions** — Sigmoid, ReLU, Tanh, Leaky ReLU
3. **2-Layer Neural Network** — Built entirely with NumPy (no TensorFlow/PyTorch)
4. **Backpropagation** — Chain rule, gradient computation, weight updates
5. **Logic Gates** — AND, OR, XOR, NAND, NOR as learning targets

## Quick Start

```bash
cd "week 7/Class 1 Project"
python3 -m venv .venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Open http://localhost:8501 if Streamlit does not open the browser automatically.

The lecture notebook is `../Class 1 Leacture/Class13_Neural_Network_Lab.ipynb`; the shorter project walkthrough is `notebooks/01_eda.ipynb`.

## Modules

| Module | What You Learn |
|--------|---------------|
| **The Neuron** | Interactive perceptron simulator with visual diagram |
| **Activation Functions** | Compare Sigmoid, ReLU, Tanh with derivatives |
| **Build Network** | Architecture builder for logic gates |
| **Training** | Live training with loss curves, decision boundaries, weight evolution |

## Key Features
- **From Scratch**: No black-box libraries. Every operation is explicit NumPy.
- **XOR Proof**: The classic demonstration that hidden layers enable non-linear learning.
- **Interactive**: Adjust weights, biases, learning rate, and hidden size in real-time.
- **Visual**: Network diagrams, decision boundaries, weight evolution plots.

## File Structure
```
neural-network-lab/
├── app.py                          # Main Streamlit app (4 modules)
├── config.py                       # Centralized configuration
├── requirements.txt                # Dependencies
├── src/
│   ├── core/
│   │   ├── neuron.py               # Single perceptron implementation
│   │   ├── activation.py           # Sigmoid, ReLU, Tanh, derivatives
│   │   └── network.py              # 2-layer NN with forward/backward/train
│   ├── gates/
│   │   └── logic_gates.py          # AND, OR, XOR, NAND, NOR datasets
│   ├── visualization/
│   │   └── nn_viz.py               # Network diagrams, decision boundaries
│   └── utils/
│       └── logger.py               # Structured logging
├── tests/
│   └── test_pipeline.py            # Unit tests (XOR requires hidden layer)
└── notebooks/
    └── 01_eda.ipynb                # Step-by-step exploration
```

## Educational Value
This project proves that neural networks are not magic — they are:
1. **Matrix multiplication** (forward pass)
2. **The chain rule from calculus** (backward pass)
3. **Gradient descent** (weight updates)

Understanding these three concepts is 90% of deep learning.

## Run the Tests

```bash
python3 -m pytest -q
python3 -m compileall -q app.py config.py src tests
```

The tests cover neuron arithmetic, activation behavior, matrix shapes, and the XOR hidden-layer requirement.
