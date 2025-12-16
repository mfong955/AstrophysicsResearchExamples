# Research Archive

This folder contains the original research code from my PhD and postdoctoral work. The code here demonstrates real-world application of data science techniques to complex scientific problems.

## Contents

| Folder | Description | Key Techniques |
|--------|-------------|----------------|
| `Astro machine learning tools/` | Core ML and statistical modeling library | MCMC, Bayesian inference, curve fitting |
| `Astro measurement examples/` | Observational data analysis | MCMC fitting, covariance handling |
| `Astro theory examples/` | Simulation data analysis | GPR, multi-dimensional binning |
| `Data replication Code Ocean/` | Reproducible research example | Docker, HDF5, data pipelines |

## Relationship to Portfolio

The portfolio projects in the parent directory were derived from this original research code:

- **[bayesian-inference/](../bayesian-inference/)** ← `Astro machine learning tools/biasProfile.py`, `Astro measurement examples/code/emcee_*.py`
- **[gaussian-process-regression/](../gaussian-process-regression/)** ← `Astro theory examples/code/z0_bin2D_plot_maps.py`
- **[statistical-modeling/](../statistical-modeling/)** ← `Astro measurement examples/code/fit_DeltaSigma.py`
- **[large-scale-data/](../large-scale-data/)** ← `Astro theory examples/code/binProfs_*.py`

See **[CODE_MAPPING.md](../CODE_MAPPING.md)** for detailed technique extraction from each file.

## Note

This code was written for scientific research and may contain domain-specific terminology. The portfolio versions have been refactored with industry-friendly naming and comprehensive documentation.
