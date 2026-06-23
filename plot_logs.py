#!/usr/bin/env python3
"""
Create visualizations for training loss and evaluation accuracy.
Generates plots from the extracted train.csv and eval.csv files.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_logs(train_csv: str = "train.csv", eval_csv: str = "eval.csv"):
    """
    Create and save plots for training loss and evaluation accuracy.
    
    Args:
        train_csv: Path to the training CSV file
        eval_csv: Path to the evaluation CSV file
    """
    # Read the CSV files
    train_df = pd.read_csv(train_csv)
    eval_df = pd.read_csv(eval_csv)
    
    # Create a figure with two subplots
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(eval_df['epoch'], eval_df['eval_accuracy'], marker='s', linestyle='-', linewidth=2, markersize=6, color='#A23B72', label='Accuracy')
    ax2.plot(eval_df['epoch'], eval_df['eval_macro_f1'], marker='^', linestyle='-', linewidth=2, markersize=6, color='#F18F01', label='Macro F1')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Evaluation Metrics over Epoch', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Combined plot saved to training_metrics.png")
    plt.close(fig)
    
    # Also create individual plots
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_df['epoch'], train_df['loss'], marker='o', linestyle='-', linewidth=2.5, markersize=5, color='#2E86AB')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss over Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training loss plot saved to training_loss.png")
    plt.close(fig1)
    
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_df['epoch'], eval_df['eval_macro_f1'], marker='^', linestyle='-', linewidth=2.5, markersize=6, color='#F18F01')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Macro F1 over Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_macro_f1.png', dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation macro F1 plot saved to evaluation_macro_f1.png")
    plt.close(fig3)
    
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_df['epoch'], eval_df['eval_accuracy'], marker='s', linestyle='-', linewidth=2.5, markersize=6, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Evaluation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Accuracy over Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✓ Evaluation accuracy plot saved to evaluation_accuracy.png")
    plt.close(fig2)


if __name__ == "__main__":
    plot_logs(train_csv="train1.csv", eval_csv="eval1.csv")
