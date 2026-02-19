"""Generate figures for paper Appendix in PDF and SVG formats."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from causal_time_prior.prior import CausalTimePrior
from causal_time_prior.visualization import (
    plot_paired_timeseries,
    plot_intervention_effect,
    plot_temporal_dag,
    plot_all_variables,
)


def main():
    """Generate paper figures in PDF and SVG formats."""
    print("=" * 80)
    print("Generating Paper Figures (PDF + SVG)")
    print("=" * 80)
    
    # Initialize prior
    print("\n1. Initializing CausalTimePrior...")
    prior = CausalTimePrior(seed=42)
    
    # Generate a representative SCM
    print("\n2. Generating temporal SCM with intervention...")
    X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
    
    # Check for divergence
    max_attempts = 10
    attempt = 0
    while (torch.isnan(X_obs).any() or torch.isnan(X_int).any()) and attempt < max_attempts:
        print(f"   Attempt {attempt + 1}: Regenerating due to NaN...")
        X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
        attempt += 1
    
    if torch.isnan(X_obs).any() or torch.isnan(X_int).any():
        print("   WARNING: Could not generate valid data after 10 attempts.")
        return
    
    print(f"   SCM: {len(scm._topo)} variables, max_lag={scm._K}")
    print(f"   Intervention: {intervention.intervention_type.value} on variables {intervention.targets}")
    
    # Create output directories
    output_dir = "demo_outputs"
    pdf_dir = os.path.join(output_dir, "pdf")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    # Generate figures in both formats
    figures = [
        ("paired_timeseries", lambda: plot_paired_timeseries(X_obs, X_int, intervention, save_path=None)),
        ("all_variables", lambda: plot_all_variables(X_obs, X_int, intervention, save_path=None)),
    ]
    
    print("\n3. Generating figures...")
    for name, plot_func in figures:
        print(f"   - {name}")
        
        # Generate plot
        plot_func()
        
        # Save as PDF (for LaTeX)
        pdf_path = os.path.join(pdf_dir, f"{name}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"     → {pdf_path}")
        
        # Save as SVG (for repo/README)
        svg_path = os.path.join(svg_dir, f"{name}.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"     → {svg_path}")
        
        plt.close()
    
    print("\n" + "=" * 80)
    print(f"Figures generated successfully!")
    print(f"  PDF: {pdf_dir}/")
    print(f"  SVG: {svg_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()