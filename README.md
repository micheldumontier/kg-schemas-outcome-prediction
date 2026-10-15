# Does the knowledge graph schema matter for clinical outcome prediction?

Code and per-fold results for a corrected and extended analysis of Jhee et al., *Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs* (ESWC 2025, LNCS 15718, pp. 282–300; [arXiv 2502.21138](https://arxiv.org/abs/2502.21138)), whose code is at [TeamHeKA/neurovasc](https://github.com/TeamHeKA/neurovasc).

The original study compared SPHN and CARE-SM knowledge graphs of synthetic patients with a ruptured intracranial aneurysm and reported large differences between schemas. This repository corrects that pipeline, repeats the comparison on the original synthetic data, and tests it on real patients with subarachnoid haemorrhage in MIMIC-IV.

## Main findings

**Synthetic data** (10-fold cross-validation, the original protocol)
- The published results reproduce, with SPHN at AUC 0.910 and CARE-SM at 0.502.
- The tabular baselines in the original study were run on a different synthetic table. On the table the graphs were built from, logistic regression reaches 0.918, level with the graph model.
- The CARE-SM result comes from pipeline defects. The preprocessing dropped all numeric values, and every feature lay four hops from the patient while the model reached three. With both corrected, CARE-SM reaches 0.821, and 0.889 with node embeddings at a usable scale.
- Age alone predicts the synthetic outcome (0.914), and the synthetic care pathways carry no outcome signal by construction, so the time encodings could not differ.

**MIMIC-IV** (1,388 patients at a 24-hour landmark, 1,287 at 48 hours and 1,190 at 72 hours)
- SPHN and CARE-SM graphs perform similarly. CARE-SM is slightly ahead of the template SPHN graph in all five landmark and seed comparisons (by 0.004 to 0.025).
- Gradient boosting on the same features matches the best graphs (0.813 at 24 hours).
- A graph built with the released SPHN 2026.1 schema and standard codes (ATC, ICD-10-PCS, LOINC, SNOMED CT) matches gradient boosting.
- Supplying ontology meaning does not change performance. Label embeddings (SapBERT), ontology embeddings (OWL2Vec*) and terminology hierarchies added as edges, from the codes themselves or from the OMOP vocabularies, all stay within 0.002 AUC of random starting vectors over three seeds.

`repro/REPORT.md` describes the reproduction in detail. The manuscript is under review; its citation will be added here.

## Contents

| Path | What it holds |
|---|---|
| `repro/` | Reproduction of the synthetic-data pipeline. `gen_caresm.py` and `preprocess_caresm.py` are ports of the original notebooks, with every change marked `CHANGED` and the literal fix behind `--fix_literals`. `run_rgcn.py` is the original R-GCN with literal encoder, generalised to any depth and resumable per fold. Also tabular baselines, diagnostics (`diag_hops.py`, `caresm_roundtrip.py`), the run registry (`runs.py`) and aggregation. |
| `repro/results/` | Per-fold metrics for every synthetic run. |
| `followup/mimic/` | MIMIC-IV v3.1 conversion from a PostgreSQL dump to Parquet without a database server (`dump_to_parquet.py`), the cohort definition (`cohort.sql`), landmark features (`landmark.py`, `build_features.py`) and an aggregate-only cohort report that suppresses counts below 11. |
| `followup/kg/` | SPHN, CARE-SM and compiled CARE-SM graph construction with pre-training checks (`build_graphs.py`), the released-SPHN graph (`build_sphn_real.py`), terminology mappings and hierarchies (`terminology.py`, `omop_ancestors.py`), label and ontology embeddings (`embed_labels.py`, `owl2vec_embed.py`, `build_ontology.py`), run queues and aggregation. |
| `followup/results/mimic24/` | Per-fold summary metrics for every MIMIC-IV run (AUC, accuracy, F1). No patient-level data and no patient counts. |
| `ontologies/fetch_sphn.sh` | Fetches the SPHN RDF Schema release (pinned to 2026.1). |

## Reproducing

**Environments.** `repro/requirements.txt` (Python 3.11) covers the synthetic pipeline and all graph training. `followup/requirements.txt` (Python 3.11) covers MIMIC-IV processing, graph building and tabular models. Label embeddings use `followup/requirements-embed.txt`, and OWL2Vec* needs its own Python 3.8 environment (`followup/requirements-owl2vec.txt`) and Java. Each directory has a `tests/` folder (`python -m pytest tests`).

**Synthetic data.** Clone [TeamHeKA/neurovasc](https://github.com/TeamHeKA/neurovasc) at commit `1619d99` into `neurovasc/` next to `repro/`. From `repro/`, generate and preprocess the CARE-SM graph (`python gen_caresm.py <csv> 10000 <out prefix>`, then `python preprocess_caresm.py <graph.nt> <out prefix> [--fix_literals]`), run the registered configurations (`python runs.py A,B,E,F,S,G,D2,E5,F5 <parallel>`), run the tabular baselines (`python tabular_baselines.py <csv>`), and aggregate (`python aggregate.py results <out dir>`). The tabular rows of the original paper reproduce only on the September 2024 version of the synthetic table in the repository history (blob `9a3f4f01`).

**MIMIC-IV.** MIMIC-IV v3.1 requires credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/) and a signed data use agreement; no data are included here. The scripts' docstrings give their arguments. The order is `dump_to_parquet.py`, `build_features.py` (with `cohort.sql`), `make_graphs.py` and `make_sphn_real.py`, then `tabular_mimic.py`, `mimic_runs.py` and `arm_runs.py`, and finally `sync_results.sh`, `aggregate_mimic.py` and `aggregate_arms.py`. Graph runs take about 0.3 to 0.7 seconds per epoch on an NVIDIA A40. Results should leave the secure environment only through `sync_results.sh`, which keeps summary metrics and drops patient counts.

**Terminologies.** ATC labels follow the WHO ATC/DDD index and ICD-10-PCS axis labels the CMS 2026 code tables. The OMOP-based hierarchy (`omop_ancestors.py`) needs an OMOP vocabulary export from [Athena](https://athena.ohdsi.org/), which includes licensed SNOMED CT and LOINC content and is not redistributed here.

## Licence

MIT (see `LICENSE`), which also carries the notice for the files adapted from TeamHeKA/neurovasc. The synthetic data and original code remain at their source under its licence.
