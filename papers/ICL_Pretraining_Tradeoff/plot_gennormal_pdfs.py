#!/usr/bin/env python3
"""
Plot Generalized Normal PDFs with different shape parameters.

This script plots the probability density functions of generalized normal distributions
with different beta (shape) parameters using jax.scipy.stats.gennorm.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.stats as stats
from scipy.special import gamma

def plot_gennormal_pdfs():
    """Plot Generalized Normal PDFs with variance-preserving scaling."""
    
    # Parameters
    x = np.linspace(-4, 4, 1000)
    loc = 0.0  # location parameter
    base_scale = 1.0  # base scale parameter
    
    # Shape parameters to plot (beta values)
    beta_values = [1.0, 1.5, 2.0]
    colors = ['red', 'blue', 'green', 'black']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    max_pdf = 0
    for beta, color, linestyle in zip(beta_values, colors, linestyles):
        # Variance-preserving scale adjustment for generalized normal
        adjusted_scale = base_scale * np.sqrt(gamma(1/beta) / gamma(3/beta))
        
        # Generalized normal distribution (JAX version doesn't use loc/scale keywords)
        # Apply location and scale transformation manually: (x - loc) / scale
        x_scaled = (x - loc) / adjusted_scale
        pdf = stats.gennorm.pdf(x_scaled, beta) / adjusted_scale
        
        if beta == 0.5:
            label = f'GenNormal (β={beta})'
        elif beta == 1.0:
            label = f'Laplace (β={beta})'
        elif beta == 2.0:
            label = f'Normal (β={beta})'
        else:
            label = f'GenNormal (β={beta})'
        
        plt.plot(x, pdf, color=color, linestyle=linestyle, linewidth=2, label=label)
        
        # Print parameters for verification
        variance = adjusted_scale**2 * gamma(3/beta) / gamma(1/beta)
        print(f"β={beta}: adjusted_scale={adjusted_scale:.4f}, variance={variance:.4f}")

        max_pdf = max(max_pdf, max(pdf))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.title('Generalized Normal PDFs with Variance-Preserving Scale\n' + 
              r'Scale = $\sqrt{\Gamma(1/\beta)/\Gamma(3/\beta)}$ for constant variance', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    plt.yscale('log')
    plt.ylim(1e-6, max_pdf * 1.1)
    
    # Add text box with explanation
    textstr = 'All distributions have unit variance\nvia scale adjustment'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('gennormal_pdfs_variance_adjusted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print("- All distributions have unit variance due to scale adjustment")
    print("- Lower β (shape parameter) → heavier tails")
    print("- β=0.5: super-exponential, β=1: Laplace, β=2: Normal, β>2: sub-Gaussian")
    print("- Scale adjustment: adjusted_scale = base_scale * sqrt(Γ(1/β)/Γ(3/β))")

if __name__ == "__main__":
    plot_gennormal_pdfs()
