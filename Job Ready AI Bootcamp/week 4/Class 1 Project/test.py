import json

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": []
}

cells = []

# Cell 1: Title & Introduction
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🧪 Class 7: Statistics for AI — Hypothesis Testing\n",
        "## A/B Test Analysis: Did the Prompt Change Actually Improve Model Accuracy?\n",
        "\n",
        "**Module Focus:**\n",
        "- **Central Limit Theorem (CLT)** — Why sample means behave predictably\n",
        "- **P-values & Significance** — Measuring evidence against a null hypothesis\n",
        "- **T-tests & ANOVA** — Comparing means between groups\n",
        "- **Correlation vs. Causation** — The critical distinction in data science"
    ]
})

# Cell 2: Setup & Imports
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "%matplotlib inline"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 3: CLT Demonstration
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔍 1. Central Limit Theorem (CLT) in Action\n",
        "Even if underlying model metrics are skewed, the distribution of **sample means** will approximate a normal distribution as sample size increases."
    ]
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "np.random.seed(42)\n",
        "# Simulate skewed accuracy scores\n",
        "population = np.random.exponential(scale=0.8, size=10000)\n",
        "\n",
        "sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(1000)]\n",
        "\n",
        "plt.figure(figsize=(8,5))\n",
        "sns.histplot(sample_means, bins=30, kde=True, color='skyblue')\n",
        "plt.title('Distribution of Sample Means (CLT)')\n",
        "plt.xlabel('Mean Accuracy')\n",
        "plt.ylabel('Frequency')\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 4: A/B Testing & T-Test
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 2. A/B Testing: Prompt A vs. Prompt B\n",
        "We'll simulate accuracy scores for two prompts and run an independent t-test."
    ]
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "prompt_a = np.random.normal(loc=0.72, scale=0.1, size=100)\n",
        "prompt_b = np.random.normal(loc=0.76, scale=0.09, size=100)\n",
        "\n",
        "t_stat, p_value = stats.ttest_ind(prompt_a, prompt_b)\n",
        "print(f'T-statistic: {t_stat:.3f} | P-value: {p_value:.4f}')\n",
        "\n",
        "if p_value < 0.05:\n",
        "    print('✅ Statistically significant! Prompt B likely performs better.')\n",
        "else:\n",
        "    print('❌ No statistically significant difference at α=0.05.')\n",
        "\n",
        "plt.figure(figsize=(8,5))\n",
        "sns.boxplot(data=[prompt_a, prompt_b], palette='pastel')\n",
        "plt.xticks([0, 1], ['Prompt A', 'Prompt B'])\n",
        "plt.ylabel('Model Accuracy')\n",
        "plt.title('A/B Test: Accuracy Distribution')\n",
        "plt.show()"
    ],
    "outputs": [],
    "execution_count": None
})

# Cell 5: ANOVA & Correlation vs Causation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📈 3. Multiple Groups & Causality\n",
        "- **ANOVA**: Use when comparing 3+ prompts/conditions.\n",
        "- **Correlation ≠ Causation**: Co-movement doesn't prove mechanism."
    ]
})

cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# One-way ANOVA\n",
        "prompt_c = np.random.normal(loc=0.78, scale=0.08, size=100)\n",
        "f_stat, anova_p = stats.f_oneway(prompt_a, prompt_b, prompt_c)\n",
        "print(f'ANOVA F={f_stat:.3f}, p={anova_p:.4f}')\n",
        "\n",
        "# Correlation example (spurious)\n",
        "np.random.seed(99)\n",
        "training_hours = np.random.uniform(10, 50, 100)\n",
        "accuracy_proxy = training_hours + np.random.normal(0, 5, 100)\n",
        "corr, p_corr = stats.pearsonr(training_hours, accuracy_proxy)\n",
        "print(f'\\nCorrelation: {corr:.3f} (p={p_corr:.4f})')\n",
        "print('⚠️ Correlation does not imply causation!')"
    ],
    "outputs": [],
    "execution_count": None
})

# Assign cells & save
notebook["cells"] = cells

with open("project.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("✅ Notebook saved as 'project.ipynb'")