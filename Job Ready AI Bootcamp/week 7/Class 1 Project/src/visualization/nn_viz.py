"""
Neural Network Visualizations
─────────────────────────────
Matplotlib-based visualizations for understanding neural networks:
- Single neuron diagrams
- Network architecture diagrams
- Decision boundary plots
- Weight evolution animations
"""

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NetworkVisualizer:
    """
    Visualization tools for neural network education.
    """

    @staticmethod
    def draw_single_neuron(
        inputs: List[float],
        weights: List[float],
        bias: float,
        z: float,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Draw a diagram of a single neuron showing inputs, weights, bias, and output.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')

        n = len(inputs)
        y_positions = np.linspace(6, 2, n) if n > 1 else [4]

        # Draw input nodes
        for i, (x_val, w_val, y_pos) in enumerate(zip(inputs, weights, y_positions)):
            # Input circle
            circle = Circle((1, y_pos), 0.4, color='#74B9FF', ec='#0984E3', linewidth=2)
            ax.add_patch(circle)
            ax.text(1, y_pos, f'x{i+1}\n{x_val:.1f}', ha='center', va='center', fontsize=10, fontweight='bold')

            # Weight label
            ax.text(2.5, y_pos + 0.3, f'w{i+1}={w_val:.2f}', fontsize=9, color='#6C5CE7')

            # Arrow to neuron
            arrow = FancyArrowPatch((1.4, y_pos), (4.2, 4), 
                                     arrowstyle='->', mutation_scale=20, 
                                     linewidth=abs(w_val)*2, color='#6C5CE7', alpha=0.7)
            ax.add_patch(arrow)

        # Bias input
        bias_y = 7 if n > 1 else 6.5
        circle_bias = Circle((1, bias_y), 0.4, color='#FDCB6E', ec='#E17055', linewidth=2)
        ax.add_patch(circle_bias)
        ax.text(1, bias_y, f'b\n{bias:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')
        arrow_bias = FancyArrowPatch((1.4, bias_y), (4.2, 4.5), 
                                      arrowstyle='->', mutation_scale=20, 
                                      linewidth=1.5, color='#E17055', alpha=0.7)
        ax.add_patch(arrow_bias)

        # Neuron body
        neuron_box = FancyBboxPatch((4.2, 3.2), 1.6, 1.6, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor='#A29BFE', edgecolor='#6C5CE7', linewidth=3)
        ax.add_patch(neuron_box)
        ax.text(5, 4.5, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        ax.text(5, 3.7, f'z={z:.2f}', ha='center', va='center', fontsize=10, color='white')

        # Activation function
        ax.text(6.5, 4.5, 'activation', fontsize=10, style='italic', color='#636E72')
        arrow_out = FancyArrowPatch((5.8, 4), (7.5, 4), 
                                     arrowstyle='->', mutation_scale=20, 
                                     linewidth=2, color='#00B894')
        ax.add_patch(arrow_out)

        # Output
        circle_out = Circle((8, 4), 0.4, color='#00B894', ec='#00B894', linewidth=2)
        ax.add_patch(circle_out)
        ax.text(8, 4, 'out', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Title
        ax.text(5, 7.5, 'Single Neuron (Perceptron)', ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def draw_network_architecture(
        input_size: int,
        hidden_size: int,
        output_size: int,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Draw a clean architecture diagram of the 2-layer network.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        layer_x = [2, 5, 8]
        layer_sizes = [input_size, hidden_size, output_size]
        layer_names = ['Input\nLayer', 'Hidden\nLayer\n(ReLU)', 'Output\nLayer\n(Sigmoid)']
        colors = ['#74B9FF', '#A29BFE', '#00B894']

        # Draw layers
        neuron_positions = []
        for x, size, name, color in zip(layer_x, layer_sizes, layer_names, colors):
            y_positions = np.linspace(8 - (size-1)*0.8, 2, size) if size > 1 else [5]
            neuron_positions.append([(x, y) for y in y_positions])

            # Layer label
            ax.text(x, 9.2, name, ha='center', va='center', fontsize=11, fontweight='bold')

            # Neurons
            for y in y_positions:
                circle = Circle((x, y), 0.35, color=color, ec='black', linewidth=1.5, zorder=3)
                ax.add_patch(circle)

        # Draw connections
        for i in range(len(neuron_positions) - 1):
            for src in neuron_positions[i]:
                for dst in neuron_positions[i + 1]:
                    ax.plot([src[0]+0.35, dst[0]-0.35], [src[1], dst[1]], 
                           'k-', alpha=0.2, linewidth=0.8, zorder=1)

        # Title
        ax.text(5, 9.8, f'Network Architecture: [{input_size} → {hidden_size} → {output_size}]', 
               ha='center', fontsize=13, fontweight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_decision_boundary(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "Decision Boundary",
        resolution: float = 0.02,
    ) -> plt.Figure:
        """
        Plot the decision boundary learned by the neural network.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))

        # Predict on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdYlBu', 
                            edgecolors='black', linewidth=1.5, s=100, zorder=3)

        # Annotate points
        for i, (xi, yi) in enumerate(X):
            ax.annotate(f'({int(xi)},{int(yi)})', (xi, yi), 
                       textcoords="offset points", xytext=(0, 10), 
                       ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Input x1')
        ax.set_ylabel('Input x2')
        ax.set_title(f'{title} — Decision Boundary Visualization', fontsize=13, fontweight='bold')
        plt.colorbar(contour, ax=ax, label='Prediction Probability')

        plt.tight_layout()
        return fig
