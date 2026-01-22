NNAST: Training Strategy & Dataset Construction (Conference-Grade Edition)

Scope: NN/GNN-based vulnerability detection for Python Web applications (Flask / Django / FastAPI)

Goal: Design a dataset and learning strategy that is credible at top-tier conferences (Security/ML), with emphasis on risk ranking, weak supervision, and reproducibility.

⸻

1. Core Philosophy

1.1 No Ground-Truth Fallacy

We explicitly do not assume the existence of ground-truth vulnerability labels.

Reasons:
	•	“Safe code” is not provable in real-world systems.
	•	Vulnerability is context- and threat-model-dependent.
	•	Binary labels introduce artificial certainty and evaluation bias.

Design choice:
	•	Replace classification with risk scoring / ranking.
	•	Treat all labels as observed signals, not truth.

⸻

2. Dataset Design Overview

2.1 Sample Unit
	•	Primary unit: Function-level code
	•	Secondary unit: Function-centered code graph
	•	AST + CFG + Data Dependency (CPG-like)

Each sample consists of:

(code_id,
 source_code,
 framework,
 graph_representation,
 weak_signals,
 meta_information)


⸻

2.2 Weak Signal Schema (Labels Are Signals)

weak_signals:
  bandit_score: float ∈ [0, 1]
  bandit_rules: list[str]
  patched_flag: bool
  synthetic_flag: bool

Important:
	•	These are not class labels.
	•	They are noisy observations used to guide ranking.

⸻

3. Data Sources (Three-Pillar Strategy)

3.1 Pillar A: Weak Supervision from Static Analysis
	•	Tool: Bandit
	•	Usage:
	•	HIGH → 1.0
	•	MEDIUM → 0.6
	•	LOW → 0.3
	•	NONE → 0.0

Design rationale:
	•	Bandit is known to produce false positives.
	•	We intentionally preserve noise to capture boundary cases.

Restrictions:
	•	Only HIGH/MEDIUM rules are used for primary signals.
	•	CWE scope limited to Web-relevant classes.

⸻

3.2 Pillar B: Real-World Patch-Diff Data

Source:
	•	Security-related commits from Python Web frameworks and applications

Construction:
	•	Pre-patch function → higher risk
	•	Post-patch function → lower risk

Notes:
	•	“Patched” does not imply “safe”
	•	Patch-diff provides relative ordering constraints, not labels

⸻

3.3 Pillar C: Controlled Synthetic Injection

Purpose:
	•	Increase coverage of rare but critical vulnerability patterns

Constraints:
	•	Only minimal AST-level modifications
	•	Only Web-relevant CWEs (e.g., injection, deserialization)
	•	Synthetic samples must be detectable by at least one static rule

Synthetic samples are always marked with synthetic_flag = true.

⸻

4. Dataset Scale Targets

Objective	Effective Samples	Feasibility
Workshop / Short	5k–10k	Very High
Top Conference	30k–100k	High
Product Prototype	100k–300k	Medium

Effective samples refer to deduplicated, non-trivial function graphs.

⸻

5. Data Collection Pipeline

Step 1: Repository Discovery
	•	GitHub Repository Search API (repo-level only)
	•	Local dependency inspection (requirements.txt / pyproject.toml)

Step 2: Local Code Extraction
	•	Clone repository
	•	Extract .py files
	•	Remove tests, vendor, generated code

Step 3: Function & Graph Extraction
	•	Parse AST
	•	Build CFG + data-flow edges
	•	Normalize identifiers

Step 4: Weak Signal Attachment
	•	Run Bandit
	•	Attach rule IDs and severity scores

Step 5: Deduplication & Filtering
	•	Hash-based deduplication
	•	Remove trivial functions (< N nodes)

⸻

6. Learning Strategy

6.1 Output Definition

The model outputs:

risk_score ∈ ℝ

No binary decision is produced by the model itself.

⸻

6.2 Training Objectives

Option A: Pointwise Regression
	•	Target: weak risk score
	•	Loss: MSE / Huber
	•	Advantage: simple, stable, fast

Option B: Pairwise Ranking (Recommended)
Training constraints:
	•	(pre-patch > post-patch)
	•	(synthetic-vuln > original)
	•	(HIGH > MEDIUM > NONE)

Loss examples:
	•	Margin ranking loss
	•	Pairwise logistic loss

⸻

7. Evaluation Protocol

7.1 Data Split
	•	Project-wise split (mandatory)
	•	No file, function, or commit leakage

Train: 60%
Valid: 20%
Test: 20%


⸻

7.2 Metrics (Ranking-Oriented)

Primary:
	•	Top-K Recall
	•	Precision@K

Secondary:
	•	AUROC
	•	AUPRC

Excluded:
	•	Accuracy
	•	F1-score

⸻

8. Baselines

Required comparisons:
	•	Rule-based: Bandit
	•	Classical ML: RF / SVM (token features)
	•	Sequence model: Transformer encoder
	•	Graph model: GNN (proposed)

⸻

9. Ablation Studies

Mandatory ablations:

Configuration	Weak	Patch	Synthetic
A	✓	✗	✗
B	✗	✓	✗
C	✗	✗	✓
Full	✓	✓	✓


⸻

10. Error Analysis & Validity

10.1 Error Analysis
	•	Common FP patterns
	•	Missed vulnerabilities (FN)
	•	Disagreement with static analyzers

10.2 Threats to Validity
	•	Weak label noise
	•	Framework bias
	•	Synthetic bias
	•	Domain shift (Web vs non-Web)

All threats are explicitly discussed.

⸻

11. Reproducibility Checklist
	•	Data collection scripts
	•	Graph construction code
	•	Fixed random seeds
	•	Public dataset release (where possible)

⸻

12. Positioning Statement (for Paper)

We propose a vulnerability risk ranking framework that abandons binary labels in favor of weakly supervised, graph-based scoring, evaluated under strict project-wise conditions on real-world Python Web applications.

⸻

13. Summary
	•	No binary vulnerability labels
	•	Weak signals treated as observations
	•	Ranking-based learning and evaluation
	•	Dataset construction is a primary contribution

This design aligns with current top-tier conference expectations in both security and machine learning research.