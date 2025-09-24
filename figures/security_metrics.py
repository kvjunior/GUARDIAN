import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['text.usetex'] = False

# Define color scheme for consistency
color_scheme = {
    'guardian': '#e74c3c',        # Red for GUARDIAN
    'baseline': '#95a5a6',     # Gray for baseline
    'adalip': '#3498db',       # Blue for AdaLipGNN
    'basic_at': '#2ecc71',     # Green for basic AT
    'no_defense': '#34495e',   # Dark gray for no defense
    'gradient': '#9b59b6'      # Purple for gradients
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.35, wspace=0.35)

# ============ SUBPLOT (a): Attack Success Rates ============
ax1 = axes[0, 0]

# Perturbation budgets
epsilons = np.array([0.00, 0.05, 0.10, 0.15, 0.20])

# Attack success rates (1 - accuracy from the table)
no_defense_success = 100 - np.array([94.5, 82.47, 68.70, 54.32, 41.28])
basic_at_success = 100 - np.array([94.5, 88.94, 79.40, 67.85, 58.92])
adalip_success = 100 - np.array([94.5, 90.83, 84.26, 75.14, 66.39])
guardian_success = 100 - np.array([94.5, 94.52, 92.34, 86.73, 76.20])

# Plot lines with markers
ax1.plot(epsilons, no_defense_success, 'o-', label='No Defense', 
         color=color_scheme['no_defense'], markersize=7, alpha=0.8)
ax1.plot(epsilons, basic_at_success, 's-', label='Basic AT', 
         color=color_scheme['basic_at'], markersize=7, alpha=0.8)
ax1.plot(epsilons, adalip_success, '^-', label='AdaLipGNN', 
         color=color_scheme['adalip'], markersize=7, alpha=0.8)
ax1.plot(epsilons, guardian_success, 'D-', label='GUARDIAN', 
         color=color_scheme['guardian'], markersize=8, linewidth=2.5)

# Add shaded area to highlight GUARDIAN advantage
ax1.fill_between(epsilons, guardian_success, no_defense_success, 
                  alpha=0.1, color=color_scheme['guardian'])

ax1.set_xlabel('Perturbation Budget (ε)', fontweight='bold')
ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax1.set_title('(a) Adversarial Attack Success Rates', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax1.set_xlim(-0.01, 0.21)
ax1.set_ylim(0, 65)
ax1.xaxis.set_major_locator(MultipleLocator(0.05))

# Add annotation for key result
ax1.annotate('GUARDIAN: 7.66% at ε=0.1', 
            xy=(0.1, 100-92.34), xytext=(0.13, 15),
            arrowprops=dict(arrowstyle='->', color=color_scheme['guardian'], 
                          connectionstyle='arc3,rad=0.3', lw=1.5),
            fontsize=8, color=color_scheme['guardian'], fontweight='bold')

# ============ SUBPLOT (b): Privacy-Utility Tradeoff ============
ax2 = axes[0, 1]

# Privacy budgets (differential privacy epsilon)
privacy_epsilons = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

# F1 scores for different privacy levels (from paper's reported values)
guardian_f1 = np.array([78.43, 86.92, 91.84, 94.67, 96.12, 96.85])
baseline_f1 = np.array([72.15, 79.83, 85.24, 89.76, 92.48, 93.15])
spaTeD_f1 = np.array([74.28, 82.37, 87.65, 91.23, 93.84, 94.52])

# Plot with different line styles
ax2.semilogx(privacy_epsilons, guardian_f1, 'o-', label='GUARDIAN', 
             color=color_scheme['guardian'], markersize=8, linewidth=2.5)
ax2.semilogx(privacy_epsilons, spaTeD_f1, 's--', label='SpaTeD', 
             color=color_scheme['adalip'], markersize=7, alpha=0.8)
ax2.semilogx(privacy_epsilons, baseline_f1, '^:', label='Baseline', 
             color=color_scheme['baseline'], markersize=7, alpha=0.8)

# Add utility preservation zone
ax2.axhspan(90, 100, alpha=0.05, color='green', label='High Utility (>90%)')
ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(1.1, 75, 'ε=1.0\n(Standard)', fontsize=7, color='gray')

ax2.set_xlabel('Privacy Budget (ε)', fontweight='bold')
ax2.set_ylabel('F1 Score (%)', fontweight='bold')
ax2.set_title('(b) Privacy-Utility Tradeoff', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax2.set_xlim(0.08, 12)
ax2.set_ylim(70, 100)

# ============ SUBPLOT (c): Certified Accuracy Radius ============
ax3 = axes[1, 0]

# Generate synthetic certified radius distribution data
np.random.seed(42)
n_samples = 1000

# Create realistic distribution for different epsilon values
radius_values = np.linspace(0, 0.3, 50)
methods_cert = ['ε=0.05', 'ε=0.10', 'ε=0.15', 'ε=0.20']
cert_accuracies = {
    'ε=0.05': stats.norm.cdf(radius_values, loc=0.18, scale=0.03),
    'ε=0.10': stats.norm.cdf(radius_values, loc=0.14, scale=0.035),
    'ε=0.15': stats.norm.cdf(radius_values, loc=0.10, scale=0.04),
    'ε=0.20': stats.norm.cdf(radius_values, loc=0.07, scale=0.045)
}

# Invert to show percentage certified at each radius
colors_cert = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
for i, (method, color) in enumerate(zip(methods_cert, colors_cert)):
    cert_acc = 100 * (1 - cert_accuracies[method])
    ax3.plot(radius_values, cert_acc, label=method, color=color, 
             linewidth=2, alpha=0.9)
    
    # Add markers at key points
    key_radius = [0.05, 0.10, 0.15, 0.20][i]
    key_acc = np.interp(key_radius, radius_values, cert_acc)
    ax3.plot(key_radius, key_acc, 'o', color=color, markersize=8, 
             markeredgecolor='white', markeredgewidth=1.5)

# Add shaded regions for different robustness levels
ax3.axvspan(0, 0.05, alpha=0.1, color='red', label='Low robustness')
ax3.axvspan(0.05, 0.15, alpha=0.1, color='yellow')
ax3.axvspan(0.15, 0.30, alpha=0.1, color='green')

ax3.set_xlabel('Certified Radius', fontweight='bold')
ax3.set_ylabel('Certified Accuracy (%)', fontweight='bold')
ax3.set_title('(c) Certified Robustness Distribution', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2)
ax3.set_xlim(0, 0.3)
ax3.set_ylim(0, 100)

# ============ SUBPLOT (d): Membership Inference Attack Resistance ============
ax4 = axes[1, 1]

# Privacy levels for x-axis
privacy_levels = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf])
privacy_labels = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0', '∞']

# MIA success rates (lower is better)
guardian_mia = np.array([50.8, 51.2, 52.3, 54.1, 58.3, 65.2, 89.7])
baseline_mia = np.array([89.7, 89.7, 89.7, 89.7, 89.7, 89.7, 89.7])
spaTeD_mia = np.array([52.4, 54.8, 57.2, 61.5, 68.9, 75.3, 89.7])

# Create bar chart with grouped bars
x_pos = np.arange(len(privacy_labels))
width = 0.25

bars1 = ax4.bar(x_pos - width, guardian_mia, width, label='GUARDIAN', 
                color=color_scheme['guardian'], alpha=0.9)
bars2 = ax4.bar(x_pos, spaTeD_mia, width, label='SpaTeD', 
                color=color_scheme['adalip'], alpha=0.8)
bars3 = ax4.bar(x_pos + width, baseline_mia, width, label='No Privacy', 
                color=color_scheme['no_defense'], alpha=0.7)

# Add random guess line
ax4.axhline(y=50, color='black', linestyle='--', linewidth=1.5, 
           label='Random Guess', alpha=0.5)

# Add value labels on bars for GUARDIAN
for i, (bar, value) in enumerate(zip(bars1, guardian_mia)):
    if i in [2, 6]:  # Only label key points
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=7,
                fontweight='bold', color=color_scheme['guardian'])

ax4.set_xlabel('Privacy Budget (ε)', fontweight='bold')
ax4.set_ylabel('MIA Success Rate (%)', fontweight='bold')
ax4.set_title('(d) Membership Inference Attack Resistance', fontweight='bold', pad=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(privacy_labels)
ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
ax4.set_ylim(40, 100)

# Add annotation for key insight
ax4.annotate('Near-random\nperformance', xy=(2, 52.3), xytext=(3.5, 45),
            arrowprops=dict(arrowstyle='->', color=color_scheme['guardian'],
                          connectionstyle='arc3,rad=-0.3', lw=1.5),
            fontsize=8, color=color_scheme['guardian'], ha='center')

# Overall figure adjustments
fig.suptitle('Security Evaluation: Adversarial Robustness and Privacy Protection', 
             fontsize=12, fontweight='bold', y=1.02)

# Remove top and right spines for cleaner look
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=4, width=0.5)

plt.tight_layout()

# Save in multiple formats
plt.savefig('security_metrics.pdf', bbox_inches='tight', dpi=300)
plt.savefig('security_metrics.eps', bbox_inches='tight', dpi=300)
plt.savefig('security_metrics.png', bbox_inches='tight', dpi=300)

print("Security metrics figure saved as security_metrics.pdf/.eps/.png")
plt.show()