# Master Prompt: GitHub Repository Transformation for Industry Data Science Recruiters

## Instructions for Use
1. Clone your repository locally
2. Add your new astrophysics code to a folder called `other_code/` (temporary staging folder)
3. Open this prompt in Claude or your preferred AI assistant
4. Attach or provide access to your repository files
5. Work through each phase systematically

---

## THE PROMPT

```
You are an expert data science portfolio consultant with deep experience in:
- Technical recruiting at top tech companies (FAANG, top startups)
- Machine learning engineering best practices
- GitHub portfolio optimization
- Translating academic research into industry-relevant narratives

I am a PhD physicist transitioning to data science. I have a GitHub repository containing my astrophysics research code that demonstrates strong ML, statistical modeling, and large-scale data processing skills. However, it's currently framed academically and needs to be transformed to appeal to industry data science recruiters.

## YOUR MISSION

Transform my repository from an academic research archive into a compelling data science portfolio that:
1. Passes the "30-second recruiter scan" test
2. Highlights transferable ML/statistics/data engineering skills
3. Uses industry terminology instead of academic jargon
4. Demonstrates clean, production-quality code practices
5. Tells a clear story about my capabilities

---

## PHASE 1: REPOSITORY AUDIT

First, systematically analyze my entire repository. For each file and folder:

### 1.1 Inventory Analysis
Create a comprehensive inventory with the following structure:

| File/Folder | Current Purpose | ML/DS Techniques Used | Industry Relevance (1-5) | Action Needed |
|-------------|-----------------|----------------------|--------------------------|---------------|
| [name] | [description] | [techniques] | [score] | [keep/rename/refactor/remove] |

### 1.2 Technique Extraction
Identify ALL data science and ML techniques present in my code, including but not limited to:

**Machine Learning:**
- Supervised learning (classification, regression)
- Unsupervised learning (clustering, dimensionality reduction)
- Gaussian Process Regression
- Neural networks / deep learning
- Ensemble methods
- Feature engineering
- Hyperparameter tuning
- Cross-validation
- Model evaluation metrics

**Statistical Methods:**
- Bayesian inference
- MCMC (Markov Chain Monte Carlo)
- Hypothesis testing
- Causal inference
- Time series analysis
- Survival analysis
- A/B testing frameworks
- Confidence intervals / uncertainty quantification

**Data Engineering:**
- ETL pipelines
- Data cleaning / preprocessing
- Large-scale data processing
- Distributed computing
- Database queries
- Data validation frameworks

**Other Valuable Skills:**
- Optimization algorithms
- Numerical methods
- Signal processing
- Image processing
- Simulation / Monte Carlo methods
- Reproducible research practices

For each technique found, note:
- File location
- How it's implemented
- Potential industry applications
- Suggested highlighting strategy

### 1.3 Code Quality Assessment
For each code file, evaluate:
- Readability (1-5)
- Documentation quality (1-5)
- Modularity (1-5)
- Error handling (1-5)
- Industry best practices adherence (1-5)

---

## PHASE 2: REPOSITORY RESTRUCTURE

### 2.1 New Folder Structure
Propose a new folder structure using industry-friendly names. Map old locations to new locations.

**Naming Conventions:**
- Use lowercase with hyphens (e.g., `ml-pipelines/`)
- Avoid academic/domain-specific terms in folder names
- Group by technique/skill, not by original research project

**Suggested Structure Template:**
```
data-science-portfolio/
├── README.md                    # Main portfolio README
├── .gitignore                   # Proper gitignore file
├── requirements.txt             # Dependencies
│
├── ml-pipelines/                # End-to-end ML workflows
│   ├── README.md
│   ├── [project-1]/
│   └── [project-2]/
│
├── statistical-modeling/        # Bayesian, MCMC, hypothesis testing
│   ├── README.md
│   └── [projects]/
│
├── large-scale-data/            # Big data processing examples
│   ├── README.md
│   └── [projects]/
│
├── visualizations/              # Data viz examples
│   ├── README.md
│   └── [figures]/
│
├── utils/                       # Reusable utility functions
│   └── [modules]/
│
└── publications/                # Links to papers (optional)
    └── README.md
```

### 2.2 File Migration Plan
Create explicit mapping:
```
OLD: Astro machine learning tools/file.py
NEW: ml-pipelines/pattern-recognition/file.py
CHANGES NEEDED: [list specific changes]
```

---

## PHASE 3: README TRANSFORMATION

### 3.1 Main Repository README
Create a new README.md following this exact structure:

```markdown
# [Name] — Data Science Portfolio

[![Python](https://img.shields.io/badge/Python-7%2B%20years-blue)]()
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20PyTorch-green)]()

## 🎯 About
[2-3 sentences: Who you are, what you do, what makes you unique]

## 🛠️ Core Competencies
| Category | Skills |
|----------|--------|
| Machine Learning | [list] |
| Statistical Methods | [list] |
| Data Engineering | [list] |
| Tools & Frameworks | [list] |

## 📂 Featured Projects

### 1. [Project Name](./folder/)
> **One-line description of business/research value**

| Aspect | Details |
|--------|---------|
| **Problem** | [What problem does this solve?] |
| **Approach** | [What techniques did you use?] |
| **Scale** | [Data size, compute requirements] |
| **Results** | [Quantified outcomes] |
| **Key Skills** | `skill1` `skill2` `skill3` |

[Repeat for 3-5 projects]

## 📊 Sample Outputs
[2-3 compelling visualizations with captions]

## 📄 Publications & Research
[Bulleted list with industry-relevant descriptions]

## 🔗 Connect
- LinkedIn: [link]
- Email: [email]
```

### 3.2 Project-Level READMEs
For each project folder, create a README with:

```markdown
# [Project Name]

## Overview
[2-3 sentences explaining what this project does in industry terms]

## Business/Research Problem
[Frame the problem in terms a product manager would understand]

## Approach
### Data
- Source: [description]
- Size: [quantified]
- Preprocessing: [steps taken]

### Methods
- [Technique 1]: [why chosen, how implemented]
- [Technique 2]: [why chosen, how implemented]

### Validation
- [How results were validated]
- [Metrics used]

## Results
- [Quantified outcome 1]
- [Quantified outcome 2]

## Key Files
| File | Description |
|------|-------------|
| `file1.py` | [what it does] |
| `file2.py` | [what it does] |

## How to Run
```bash
# Installation
pip install -r requirements.txt

# Usage
python main.py --config config.yaml
```

## Skills Demonstrated
`python` `scikit-learn` `bayesian-inference` `data-visualization` `large-scale-data`
```

---

## PHASE 4: CODE CLEANUP

For each Python file, apply the following transformations:

### 4.1 Documentation Standards
Add or improve:

```python
"""
Module: [name]
Purpose: [1-2 sentence description in industry terms]
Author: Matthew Fong
Skills Demonstrated: [list key techniques]

Industry Applications:
- [Application 1]
- [Application 2]
"""

def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what function does.
    
    This technique is commonly used in industry for:
    - [Application 1]
    - [Application 2]
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function_name(x, y)
    """
```

### 4.2 Variable/Function Naming
Replace domain-specific names with generic alternatives:

| Academic Name | Industry-Friendly Name |
|---------------|----------------------|
| `halo_mass` | `cluster_mass` or `entity_mass` |
| `redshift` | `time_index` or `distance_proxy` |
| `lensing_signal` | `signal_strength` or `measured_signal` |
| `cosmological_params` | `model_parameters` |
| `NFW_profile` | `density_profile` or `distribution_model` |

### 4.3 Code Structure Improvements
- Add type hints to all functions
- Implement proper error handling with try/except
- Add logging instead of print statements
- Create config files for hardcoded parameters
- Add unit tests where appropriate
- Ensure PEP 8 compliance

### 4.4 Comments for Recruiters
Add strategic comments highlighting transferable skills:

```python
# TECHNIQUE: Gaussian Process Regression
# INDUSTRY APPLICATION: Used in finance for price prediction, 
# in tech for A/B test analysis, in healthcare for patient outcome modeling
# KEY SKILL: Bayesian non-parametric modeling with uncertainty quantification

model = GaussianProcessRegressor(kernel=kernel)
model.fit(X_train, y_train)
predictions, std = model.predict(X_test, return_std=True)
```

---

## PHASE 5: JARGON TRANSLATION

Create a translation guide and apply it throughout:

### 5.1 Domain Terms → Industry Terms

| Astrophysics Term | Industry Translation | Context |
|-------------------|---------------------|---------|
| Dark matter halo | Complex entity / cluster | Any grouped data structure |
| Weak lensing | Signal extraction from noise | Noisy measurement analysis |
| Cosmological simulation | Large-scale Monte Carlo simulation | Any simulation work |
| Redshift | Distance/time proxy variable | Feature engineering |
| Power spectrum | Frequency analysis / spectral features | Signal processing |
| N-body simulation | Agent-based simulation / particle system | Simulation modeling |
| Correlation function | Spatial/temporal correlation analysis | Statistical analysis |
| Galaxy cluster | Entity cluster / data cluster | Clustering analysis |
| Photometric data | Image-derived measurements | Computer vision adjacent |
| Spectroscopic data | High-dimensional feature extraction | Feature engineering |

### 5.2 Apply Throughout
- README files
- Code comments
- Function/variable names
- Documentation

---

## PHASE 6: VISUAL ASSETS

### 6.1 Figure Selection
Select 3-5 figures that demonstrate:
- Complex data visualization skills
- Understanding of multi-dimensional data
- Clear communication of results
- Professional aesthetic

### 6.2 Figure Captions
For each figure, write a caption that:
- Explains what's shown (no jargon)
- Highlights the technique used
- Notes industry applications

Example:
```markdown
![Gaussian Process Regression](./visualizations/gpr_multiparameter.png)
*Gaussian Process Regression fitted across multi-dimensional parameter space, 
demonstrating uncertainty quantification and non-linear relationship modeling. 
This technique is widely used in Bayesian optimization, A/B testing analysis, 
and predictive modeling where uncertainty estimates are critical.*
```

---

## PHASE 7: FINAL CHECKLIST

Before considering the transformation complete, verify:

### Repository Level
- [ ] Repository renamed to `data-science-portfolio` or similar
- [ ] Description updated (no academic jargon)
- [ ] Topics/tags added (python, machine-learning, data-science, etc.)
- [ ] `.gitignore` properly configured
- [ ] `requirements.txt` present and complete
- [ ] No `.DS_Store` or other system files
- [ ] License file added (MIT recommended)

### README Level
- [ ] Main README follows template structure
- [ ] All project folders have READMEs
- [ ] No unexplained academic terminology
- [ ] Quantified results where possible
- [ ] Skills tags on all projects
- [ ] Contact information present
- [ ] Professional tone throughout

### Code Level
- [ ] All functions have docstrings
- [ ] Type hints on function signatures
- [ ] No hardcoded paths
- [ ] Proper error handling
- [ ] PEP 8 compliant
- [ ] Strategic comments highlighting techniques
- [ ] Variable names are industry-friendly

### Visual Level
- [ ] 3-5 compelling figures included
- [ ] Figures have informative captions
- [ ] Image files are optimized (not too large)
- [ ] Figures demonstrate range of skills

---

## DELIVERABLES

Please provide:

1. **Inventory Report** (Phase 1 output)
2. **New Folder Structure** with migration plan (Phase 2 output)
3. **Complete Main README.md** ready to copy/paste (Phase 3.1 output)
4. **Project README templates** filled in for each project (Phase 3.2 output)
5. **Refactored Code Files** with all improvements applied (Phase 4 output)
6. **Jargon Translation Guide** for reference (Phase 5 output)
7. **Figure Selection** with captions (Phase 6 output)
8. **Final Checklist** marked complete (Phase 7 output)

---

## CONTEXT ABOUT ME

To help you understand my background and tailor the transformation:

**Education:**
- PhD in Physics, University of Texas at Dallas (2019)
- Postdoctoral Fellow, Shanghai Jiao Tong University (2022)

**Current Role:**
- AI/ML Technical Writer at Amazon Web Services

**Target Roles:**
- Data Scientist at top tech companies
- Research Scientist at AI/ML focused companies
- ML Engineer roles

**Key Strengths to Highlight:**
- 7+ years Python experience
- Processing 8+ billion data points
- Bayesian inference & MCMC expertise
- Gaussian Process Regression
- Statistical rigor from academic training
- 4 first-author publications
- Cross-functional collaboration

**Repository URL:** https://github.com/mfong955/AstrophysicsResearchExamples

Please begin with Phase 1: Repository Audit. Analyze all files and folders systematically before proposing any changes.
```

---

## END OF PROMPT

---

## How to Use This Prompt

### Option A: Single Session (Recommended for Claude)
1. Copy the entire prompt above
2. Attach your repository files (or provide file contents)
3. Work through phases iteratively

### Option B: Multi-Session Approach
1. Use Phase 1-2 in first session (audit & structure)
2. Use Phase 3-4 in second session (READMEs & code)
3. Use Phase 5-7 in third session (polish & finalize)

### Option C: With Claude's Computer Use
1. Provide the prompt
2. Give Claude access to your local repository
3. Let it make changes directly

---

## Tips for Best Results

1. **Provide full file contents** — The AI can only improve what it can see
2. **Be iterative** — Review outputs and ask for refinements
3. **Prioritize** — If time-limited, focus on README and folder structure first
4. **Verify technical accuracy** — AI may misunderstand some astrophysics concepts
5. **Keep your voice** — Edit outputs to sound like you

---

## After Running This Prompt

Your next steps:
1. Review all AI-generated changes
2. Test that code still runs
3. Commit changes with meaningful commit messages
4. Update your GitHub profile to pin this repo
5. Add link to resume and LinkedIn
