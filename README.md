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
- With the corrected resampled t-test (Nadeau and Bengio), which allows for the overlap between training folds, no graph differs significantly from gradient boosting or from another schema in the full cohort.
- CARE-SM needs four layers, or shortcuts that bring values within reach of three. At three layers the original CARE-SM graph is at chance (0.501), whereas SIO *has attribute* and feature-specific attribute shortcuts reach 0.801 and 0.810; at four layers all matched variants lie between 0.803 and 0.814.
- Event timing adds nothing beyond occurrence. Care-event occurrence alone gives 0.723 at 24 hours, adding first times lowers it at every landmark, and pairwise order, intervals and clinical timing such as the time to aneurysm repair do not raise it.
- Standardising literals within training folds changes graph AUC by at most 0.006. CARE-SM is well calibrated, SPHN somewhat overconfident and untuned gradient boosting strongly overconfident, and after Platt recalibration gradient boosting matches CARE-SM (Brier score 0.463 each).
- In the 432 patients with aneurysm coiling or clipping, tabular models keep their AUC (0.801 to 0.806) whereas the CARE-SM (0.765) and SPHN (0.718) graphs fall.

The project page at https://micheldumontier.github.io/kg-schemas-outcome-prediction/ presents the results, with every number computed from the per-fold results by `site/build.py` (`python site/build.py` rewrites `docs/index.html`). The page covers the main comparisons with plain paired t-tests and does not yet show the analyses added for the camera-ready paper, which the table below locates. `repro/REPORT.md` describes the reproduction in detail. The paper, *Does the Knowledge Graph Schema Matter for Clinical Outcome Prediction?*, is to appear at MIE 2027, and its citation will be added here.

## Where each result of the MIE 2027 paper comes from

Paths are relative to the repository root. Every results folder holds aggregate or per-fold summary metrics only.

| Result in the paper | Script | Results |
|---|---|---|
| Table 2, synthetic AUC and macro-F1 | `repro/runs.py`, `repro/aggregate.py`, `repro/tabular_baselines.py` | `repro/results/` |
| Table 2, MIMIC-IV AUC at 24, 48 and 72 h | `followup/kg/mimic_runs.py`, `followup/kg/tabular_mimic.py`, `followup/kg/aggregate_mimic.py` | `followup/results/mimic24/` |
| Macro-F1 in Table 2 | `followup/kg/f1_macros.py` | `f1_macro` in the per-fold files of `repro/results/` and `followup/results/mimic24/` |
| Corrected resampled t-tests and 95% intervals | `followup/kg/corrected_stats.py` | `followup/results/corrected_stats.md` and `.json` |
| Occurrence versus first times | `followup/kg/timing_mimic.py`, with the corrected test in `followup/kg/timing_order.py` | `followup/results/timing/`, and the `EV_OCC_TIME - EV_OCC` rows of `followup/results/timing_order/summary.json` |
| Timing order, intervals and clinical timing | `followup/kg/timing_order.py` | `followup/results/timing_order/` |
| CARE-SM SIO and feature-specific shortcut variants | `followup/kg/caresm_variants.py`, `followup/kg/aggregate_caresm_variants.py` | `followup/results/caresm_variants/`, with corrected tests in `followup/results/corrected_stats.md` |
| Released-term SPHN 2026.1 graph at 24, 48 and 72 h | `followup/kg/build_sphn_real.py`, `followup/kg/make_sphn_real.py`, the flags of `followup/kg/pilot_runs.sh` | `followup/results/mimic24/MIMIC{24,48,72}_sphn_real/` |
| Label, ontology and hierarchy enrichments | `followup/kg/arm_runs.py`, `followup/kg/aggregate_arms.py` | `followup/results/mimic24/MIMIC24s{0,1,2}_arm_*/` |
| Fold-wise standardisation | `repro/foldstd.py` (`run_rgcn.py --foldwise_std`), `followup/kg/foldwise_runs.sh`, `followup/kg/aggregate_foldwise.py` | `followup/results/foldwise/` |
| Calibration (slopes, Platt-recalibrated gradient boosting, Brier score, ECE) | `repro/calibration.py` (`run_rgcn.py` and `tabular_mimic.py` with `--calibration` and `--save_probs`), `followup/kg/calibration_slopes.py` | `followup/results/calibration/`, with per-fold Brier scores in `followup/results/foldwise/` |
| Aneurysm-repair cohort | `followup/kg/make_aneurysm_subset.py`, `followup/kg/cohort_runs.py`, `followup/kg/aggregate_cohort.py` | `followup/results/cohort_aneurysm/` |

## Contents

| Path | What it holds |
|---|---|
| `repro/` | Reproduction of the synthetic-data pipeline. `gen_caresm.py` and `preprocess_caresm.py` are ports of the original notebooks, with every change marked `CHANGED` and the literal fix behind `--fix_literals`. `run_rgcn.py` is the original R-GCN with literal encoder, generalised to any depth and resumable per fold. Also tabular baselines, diagnostics (`diag_hops.py`, `caresm_roundtrip.py`), the run registry (`runs.py`) and aggregation. |
| `repro/results/` | Per-fold metrics for every synthetic run. |
| `followup/mimic/` | MIMIC-IV v3.1 conversion from a PostgreSQL dump to Parquet without a database server (`dump_to_parquet.py`), the cohort definition (`cohort.sql`), landmark features (`landmark.py`, `build_features.py`) and an aggregate-only cohort report that suppresses counts below 11. |
| `followup/kg/` | SPHN, CARE-SM and compiled CARE-SM graph construction with pre-training checks (`build_graphs.py`), the matched CARE-SM variants (`caresm_variants.py`), the released-SPHN graph (`build_sphn_real.py`), terminology mappings and hierarchies (`terminology.py`, `omop_ancestors.py`), label and ontology embeddings (`embed_labels.py`, `owl2vec_embed.py`, `build_ontology.py`), timing analyses (`timing_mimic.py`, `timing_order.py`), the aneurysm-repair subset (`make_aneurysm_subset.py`), calibration (`calibration_slopes.py`), corrected tests (`corrected_stats.py`), run queues and aggregation. |
| `followup/results/mimic24/` | Per-fold summary metrics for every MIMIC-IV run of the main comparison (AUC, accuracy, F1). No patient-level data and no patient counts. |
| `followup/results/` (other folders) | Per-fold metrics and summaries of the timing, CARE-SM variant, fold-wise standardisation, calibration and aneurysm-cohort analyses, and the corrected tests (`corrected_stats.md`). Counts below 11 are suppressed and per-fold files carry no patient counts. |
| `ontologies/fetch_sphn.sh` | Fetches the SPHN RDF Schema release (pinned to 2026.1). |

## Reproducing

**Environments.** `repro/requirements.txt` (Python 3.11) covers the synthetic pipeline and all graph training. `followup/requirements.txt` (Python 3.11) covers MIMIC-IV processing, graph building and tabular models. Label embeddings use `followup/requirements-embed.txt`, and OWL2Vec* needs its own Python 3.8 environment (`followup/requirements-owl2vec.txt`) and Java. Each directory has a `tests/` folder (`python -m pytest tests`).

**Synthetic data.** Clone [TeamHeKA/neurovasc](https://github.com/TeamHeKA/neurovasc) at commit `1619d99` into `neurovasc/` next to `repro/`. From `repro/`, generate and preprocess the CARE-SM graph (`python gen_caresm.py <csv> 10000 <out prefix>`, then `python preprocess_caresm.py <graph.nt> <out prefix> [--fix_literals]`), run the registered configurations (`python runs.py A,B,E,F,S,G,D2,E5,F5 <parallel>`), run the tabular baselines (`python tabular_baselines.py <csv>`), and aggregate (`python aggregate.py results <out dir>`). The tabular rows of the original paper reproduce only on the September 2024 version of the synthetic table in the repository history (blob `9a3f4f01`).

**MIMIC-IV.** MIMIC-IV v3.1 requires credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/) and a signed data use agreement; no data are included here. The scripts' docstrings give their arguments. The order is `dump_to_parquet.py`, `build_features.py` (with `cohort.sql`), `make_graphs.py` and `make_sphn_real.py`, then `tabular_mimic.py`, `mimic_runs.py` and `arm_runs.py`, and finally `sync_results.sh`, `aggregate_mimic.py` and `aggregate_arms.py`. The further analyses reuse these tables and graphs. The CARE-SM variants run with `caresm_variants.py build` and `queue`, followed by `aggregate_caresm_variants.py`. The timing analyses are `timing_mimic.py` and `timing_order.py`. Fold-wise standardisation and calibration use `foldwise_runs.sh` and the tabular commands in its header, followed by `aggregate_foldwise.py` and `calibration_slopes.py`. The aneurysm cohort runs `make_aneurysm_subset.py`, then `make_graphs.py` and `tabular_mimic.py` on the subset, `cohort_runs.py` and `aggregate_cohort.py`. Last, `corrected_stats.py` and `f1_macros.py` read only the results in this repository. Scripts that import from `repro/` run from the repository root with `PYTHONPATH=repro`. Graph runs take about 0.3 to 0.7 seconds per epoch on an NVIDIA A40. Results should leave the secure environment only through `sync_results.sh` or the aggregation scripts of the further analyses, which keep summary metrics, drop patient counts and suppress counts below 11.

**Terminologies.** ATC labels follow the WHO ATC/DDD index and ICD-10-PCS axis labels the CMS 2026 code tables. The OMOP-based hierarchy (`omop_ancestors.py`) needs an OMOP vocabulary export from [Athena](https://athena.ohdsi.org/), which includes licensed SNOMED CT and LOINC content and is not redistributed here.

## Licence

MIT (see `LICENSE`), which also carries the notice for the files adapted from TeamHeKA/neurovasc. The synthetic data and original code remain at their source under its licence.
