import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8

# Create figure with 2x2 subplots
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ============ SUBPLOT (a): Transfer Matrix ============
ax1 = fig.add_subplot(gs[0, 0])

# Transfer matrix data (F1 scores from paper)
datasets = ['ETH-S', 'ETH-P', 'BTC-M', 'BTC-L']
transfer_matrix = np.array([
    [97.83, 89.42, 78.36, 82.94],  # ETH-S trained
    [88.76, 95.26, 76.83, 81.25],  # ETH-P trained
    [74.28, 72.94, 93.17, 87.63],  # BTC-M trained
    [79.85, 77.16, 89.28, 98.42]   # BTC-L trained
])

# Create custom colormap
colors = ['#fff7fb', '#ece2f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016450']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create heatmap
im = ax1.imshow(transfer_matrix, cmap=cmap, vmin=70, vmax=100, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('F1 Score (%)', rotation=270, labelpad=15)

# Set ticks and labels
ax1.set_xticks(np.arange(len(datasets)))
ax1.set_yticks(np.arange(len(datasets)))
ax1.set_xticklabels(datasets)
ax1.set_yticklabels(datasets)
ax1.set_xlabel('Test Dataset', fontweight='bold')
ax1.set_ylabel('Train Dataset', fontweight='bold')

# Rotate the tick labels and set their alignment
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(datasets)):
    for j in range(len(datasets)):
        value = transfer_matrix[i, j]
        color = 'white' if value < 85 else 'black'
        text = ax1.text(j, i, f'{value:.1f}', ha="center", va="center", 
                       color=color, fontsize=8, fontweight='bold')
        
        # Highlight diagonal (same dataset)
        if i == j:
            rect = Rectangle((j-0.45, i-0.45), 0.9, 0.9, 
                           fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)

ax1.set_title('(a) Cross-Blockchain Transfer Matrix', fontweight='bold', pad=10)

# ============ SUBPLOT (b): Performance Degradation ============
ax2 = fig.add_subplot(gs[0, 1])

categories = ['Within\nEthereum', 'Within\nBitcoin', 'ETH→BTC', 'BTC→ETH']
baseline_performance = [96.5, 95.8, 96.5, 95.8]  # Original performance
transfer_performance = [89.42, 87.63, 77.60, 78.57]  # After transfer
degradation = [(b - t) for b, t in zip(baseline_performance, transfer_performance)]

x_pos = np.arange(len(categories))
width = 0.35

# Create grouped bars
bars1 = ax2.bar(x_pos - width/2, baseline_performance, width, 
                label='Original', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, transfer_performance, width,
                label='Transfer', color='#e74c3c', alpha=0.8)

# Add degradation percentage labels
for i, (b, t, d) in enumerate(zip(baseline_performance, transfer_performance, degradation)):
    ax2.annotate(f'-{d:.1f}%', xy=(i, t), xytext=(i, t-5),
                ha='center', fontsize=7, color='darkred', fontweight='bold')
    
    # Add connecting lines
    ax2.plot([i-width/2, i+width/2], [b, t], 'k--', alpha=0.3, linewidth=0.8)

ax2.set_ylabel('F1 Score (%)', fontweight='bold')
ax2.set_xlabel('Transfer Type', fontweight='bold')
ax2.set_title('(b) Performance Degradation by Protocol', fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim(70, 100)

# Add threshold line
ax2.axhline(y=85, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.text(3.5, 85.5, 'R4: 15% threshold', fontsize=7, color='green', ha='right')

# ============ SUBPLOT (c): Feature Importance ============
ax3 = fig.add_subplot(gs[1, 0])

features = ['Transaction\nVolume', 'Temporal\nPatterns', 'Network\nTopology', 
            'Gas/Fee\nStructure', 'Contract\nInteractions']
ethereum_importance = [0.23, 0.18, 0.21, 0.15, 0.23]
bitcoin_importance = [0.28, 0.22, 0.25, 0.20, 0.05]

x = np.arange(len(features))
width = 0.35

# Create bars
bars1 = ax3.bar(x - width/2, ethereum_importance, width, 
                label='Ethereum', color='#627eea', alpha=0.9)
bars2 = ax3.bar(x + width/2, bitcoin_importance, width,
                label='Bitcoin', color='#f7931a', alpha=0.9)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

ax3.set_ylabel('Feature Importance Score', fontweight='bold')
ax3.set_xlabel('Feature Type', fontweight='bold')
ax3.set_title('(c) Feature Importance Across Blockchains', fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(features, fontsize=7)
ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_ylim(0, 0.35)

# Highlight key differences
ax3.annotate('Protocol-specific', xy=(4, 0.23), xytext=(3.5, 0.30),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=8, color='red', fontweight='bold')

# ============ SUBPLOT (d): Domain Adaptation ============
ax4 = fig.add_subplot(gs[1, 1])

epochs = np.array([0, 1, 2, 3, 5, 10])
scenarios = {
    'Zero-shot': [78.3, 78.3, 78.3, 78.3, 78.3, 78.3],
    'Fine-tune (1% data)': [78.3, 82.1, 84.5, 86.2, 87.8, 88.5],
    'Fine-tune (5% data)': [78.3, 84.7, 87.2, 89.1, 90.8, 91.6],
    'Fine-tune (10% data)': [78.3, 86.3, 89.4, 91.2, 92.7, 93.4],
    'Full retrain': [78.3, 88.9, 91.8, 93.5, 94.8, 95.2]
}

colors_adapt = ['#34495e', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
linestyles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', '^', 'D', 'v']

for (scenario, values), color, ls, marker in zip(scenarios.items(), colors_adapt, 
                                                  linestyles, markers):
    ax4.plot(epochs, values, label=scenario, color=color, linestyle=ls, 
            marker=marker, markersize=6, linewidth=1.5, alpha=0.9)

# Add shaded region for acceptable performance
ax4.axhspan(85, 100, alpha=0.05, color='green')
ax4.text(8, 86, 'Acceptable\nPerformance', fontsize=8, color='green', alpha=0.7)

ax4.set_xlabel('Fine-tuning Epochs', fontweight='bold')
ax4.set_ylabel('F1 Score (%)', fontweight='bold')
ax4.set_title('(d) Domain Adaptation Effectiveness', fontweight='bold', pad=10)
ax4.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=7)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(-0.5, 10.5)
ax4.set_ylim(75, 98)

# Add annotation for key finding
ax4.annotate('5% data reaches\n90% performance', 
            xy=(5, 90.8), xytext=(7, 82),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color='#2ecc71', lw=1.5),
            fontsize=8, color='#2ecc71', fontweight='bold')

# Overall figure adjustments
fig.suptitle('Cross-Blockchain Transfer Learning Analysis', 
             fontsize=12, fontweight='bold', y=1.00)

# Remove top and right spines
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=0.5)

plt.tight_layout()

# Save in multiple formats
plt.savefig('transfer_analysis.pdf', bbox_inches='tight', dpi=300)
plt.savefig('transfer_analysis.eps', bbox_inches='tight', dpi=300)
plt.savefig('transfer_analysis.png', bbox_inches='tight', dpi=300)

print("Transfer analysis figure saved as transfer_analysis.pdf/.eps/.png")
plt.show()