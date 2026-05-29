# 🧠 Neural Network Lab

A production-grade, 100% offline educational platform for learning Neural Networks from scratch.

## Topics Covered
1. **The Neuron** — Weights, biases, and the perceptron model
2. **Activation Functions** — Sigmoid, ReLU, Tanh, Leaky ReLU
3. **2-Layer Neural Network** — Built entirely with NumPy (no TensorFlow/PyTorch)
4. **Backpropagation** — Chain rule, gradient computation, weight updates
5. **Logic Gates** — AND, OR, XOR, NAND, NOR as learning targets

## Quick Start

```bash
cd neural-network-lab
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

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
