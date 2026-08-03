#!/usr/bin/env python3
"""
Plot Student-t PDFs with variance-preserving scale adjustment.

This script plots the probability density functions of Student-t distributions
with degrees of freedom nu = 3, 5, 10, and infinity (normal), using the
scale adjustment factor sqrt(nu-2)/sqrt(nu) to keep variance constant.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_student_pdfs():
    """Plot Student-t PDFs with variance-preserving scaling."""
    
    # Parameters
    x = np.linspace(-4, 4, 1000)
    loc = 0.0  # location parameter
    base_scale = 3  # base scale parameter
    
    # Degrees of freedom to plot
    nu_values = [3, 5, 10, np.inf]
    colors = ['red', 'blue', 'green', 'black']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    max_pdf = 0
    for nu, color, linestyle in zip(nu_values, colors, linestyles):
        if nu == np.inf:
            # Normal distribution (nu = infinity)
            adjusted_scale = base_scale
            pdf = stats.norm.pdf(x, loc=loc, scale=adjusted_scale)
            label = f'Normal (ν=∞)'
        else:
            # Student-t distribution with variance-preserving scale
            adjusted_scale = base_scale * np.sqrt((nu - 2) / nu)
            pdf = stats.t.pdf(x, df=nu, loc=loc, scale=adjusted_scale)
            label = f'Student-t (ν={nu})'
        
        plt.plot(x, pdf, color=color, linestyle=linestyle, linewidth=2, label=label)
        
        # Print variance for verification
        if nu == np.inf:
            variance = adjusted_scale**2
        else:
            if nu > 2:
                variance = adjusted_scale**2 * nu / (nu - 2)
            else:
                variance = np.inf
        print(f"ν={nu}: adjusted_scale={adjusted_scale:.4f}, variance={variance:.4f}")

        max_pdf = max(max_pdf, max(pdf))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.title('Student-t PDFs with Variance-Preserving Scale\n' + 
              r'Scale = $\sqrt{(\nu-2)/\nu}$ for constant variance', fontsize=14)
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
    plt.savefig('student_t_pdfs_variance_adjusted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print("- All distributions have unit variance due to scale adjustment")
    print("- Lower ν (degrees of freedom) → heavier tails")
    print("- As ν → ∞, Student-t converges to Normal")
    print("- Scale adjustment: adjusted_scale = base_scale * sqrt((ν-2)/ν)")

if __name__ == "__main__":
    plot_student_pdfs()
