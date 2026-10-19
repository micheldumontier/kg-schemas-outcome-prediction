# Reproduction of Jhee et al., ESWC 2025 — with a focus on the CARE-SM graph

Paper: *Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs*,
LNCS 15718, pp. 282–300 (arXiv 2502.21138v2; published version is textually identical apart from the data-access statement).
Code: https://github.com/TeamHeKA/neurovasc (commit `1619d99`).

Sections 1 to 5 were run on one fold (fold 0) of the paper's own 10-fold split (`StratifiedKFold(10, shuffle, random_state=77)` + 1/9 validation),
with the repo's RGCN+lit model unchanged (3 layers unless stated, embed/hidden 32, lr 1e-3, wd 5e-4, dropout 0.2, best-val-accuracy checkpoint).
Epoch budgets are below the paper's 2000 (CPU only) — see each row.

## Current results (10 folds, 14 October 2026)

Sections 1 to 5 below record the initial single-fold exploration. The full 10-fold runs (2,000 epochs, the paper's
protocol) supersede its numbers. Two single-fold results changed materially and are corrected here.

| Run | Description | 10-fold AUC | Single fold (below) |
|---|---|---|---|
| A | CARE-SM-ts as published, 3 layers | 0.502 ± 0.011 | 0.494 |
| B | CARE-SM-ts as published, 5 layers | 0.499 ± 0.018 | stopped at chance |
| E | literal predicate fixed, 4 layers | 0.821 ± 0.015 | 0.809 |
| E5 | literal predicate fixed, 5 layers | 0.833 ± 0.013 | |
| F | E with unit-scale embeddings | 0.889 ± 0.008 | 0.882 |
| F5 | E5 with unit-scale embeddings | 0.890 ± 0.008 | |
| S | SPHN-ts | 0.910 ± 0.007 | 0.918 |
| G | SPHN-ts with `hasAge` relabelled `hasDiagnosis` | **0.843 ± 0.068** | 0.678 |
| D2 | CARE-SM-ts with a `hasOutcome` edge per patient | 0.981 ± 0.040 | 0.955 |
| LR / RF / NN | same table as the graphs | 0.918 / 0.916 / 0.912 | |
| LR / RF / NN | Sept-2024 table (paper's tabular rows) | 0.707 / 0.708 / 0.702 | |
| Age only (RF) | | 0.914 ± 0.007 | |
| Events only (RF) | | 0.504 ± 0.019 | |
| Round trip (RF on features read back from CARE-SM) | | 0.917 ± 0.006 | |

**Corrections to the conclusions below.**
- The extra relation in the paper's Table 5 statistics is the outcome triple counted in the statistics, not an edge in
  the training graph (see the updated conclusion in section 4).
- The `hasAge` ablation (G) lowers SPHN to 0.843 over 10 folds, not 0.678. The dedicated predicate explains part of
  the SPHN advantage, not most of it, and the effect varies between folds (standard deviation 0.068).
- With five layers the corrected CARE-SM model reaches 0.833 (E5), so the original authors' five-layer failure came from
  the dropped literals rather than from depth.

**Real data.** The follow-up on MIMIC-IV (1,388 patients with subarachnoid haemorrhage at a 24-hour landmark, 1,287 at
48 hours and 1,190 at 72 hours) is in `followup/` and in the MIE 2027 manuscript. In short, CARE-SM and SPHN graphs
performed similarly with a small, consistent CARE-SM advantage; gradient boosting matched the best graphs; a graph with
released SPHN 2026.1 terms matched gradient boosting; and SapBERT label embeddings did not help. Per-fold metrics are in
`followup/results/mimic24/`.

## 1. Which data the published graphs were built from

| Evidence | Result |
|---|---|
| Committed processed SPHN graph (`notebooks/Graphs/processed_data`) — patient ages read back from triples | 100 % match `data/syn_data.csv` (strong-signal CSV); ~2 % match any other historical CSV |
| CARE-SM graph regenerated from `data/syn_data.csv` with the repo notebook | 1,744,505 triples (+ `time:before` → 1,770,497) |
| Same notebook on the Sept-2024 CSV (blob `9a3f4f01`, the one shown in the notebooks' saved outputs) | 1,744,065 triples |
| Paper Table 5, CARE-SM\* | 1,754,505 triples, 13 relations, 576,733 entities, 24,766 literals |

The paper's CARE-SM\*-ts graph = our graph from `data/syn_data.csv` **+ exactly 10,000 triples of one extra relation**
(adding one `nvasc:hasOutcome` triple per patient reproduces triples = 1,754,505 and relations = 13 exactly;
entities − timestamp literals = 576,730 vs 576,733 and literals 24,723 vs 24,766 — differences fit unseeded timestamp collisions).

On `data/syn_data.csv`, with the identical patient order and split: **RF AUC 0.914 / LR 0.918 (fold 0); 10-fold 0.916 / 0.918.**
**Age alone gives RF AUC 0.915.** The paper's tabular numbers (RF 0.71) reproduce only on the Sept-2024 CSV (RF 0.709).

## 2. Defects in the CARE-SM production / preprocessing

The CARE-SM graph itself is **lossless**: reading each patient's 30 features back from the triples alone
(patient → role → process → output → class / `has-value`) and fitting RF gives **AUC 0.917 ± 0.006**, the same as the CSV.
The failures are in how the graph is turned into model input and in the model's reach:

1. **Numeric literals silently dropped.** `data_preprocess_CARESM.ipynb` selects literals with
   `r == '<http://sphn.org/has-value>'`; the CARE-SM graph uses `sio:has-value`. 0 of 40,000 value triples match, so
   age / GCS / length of stay / number of acts never reach the model; only timestamps do. (Had the predicate matched, the
   un-assigned `removesuffix` would have made `pd.to_numeric` fail on `"23.0^^<xsd:float>"`.) Table 5's "24,766 literals"
   equals the timestamp-only count, i.e. the published run had this bug.
2. **Numeric features minted as per-patient classes in the NCIt namespace**: `ncit:hospital_stay_length_13.5956733…`
   (10,000 distinct classes for stay, GCS; 9,999 for acts). Without the literal values these carry no shared signal. Every
   feature value class (`NCIT_diabete_1.0`, …) is a fabricated IRI inside `http://purl.obolibrary.org/obo/NCIT_`.
3. **Every feature is 4 message-passing hops from the patient** (patient ← role ← process ← output ← value class/literal),
   measured on all 22 features and all literals. The paper's RGCN3+lit has a 3-hop receptive field, so it is
   **structurally blind to all patient features** — chance-level AUC is guaranteed, not a finding about the schema.
   In SPHN the same features are 2 hops away (literals 2–3).
4. **All CARE-SM features share the same predicates** (`is-realized-in` → `has-output` → `rdf:type`/`has-value`), so after mean
   aggregation at the single role node the model cannot tell age's value from GCS's. Feature identity lives only in node
   identity, and node embeddings are xavier-initialised over ~600k rows (|x| ≤ 0.0045) and **never trained** (the
   `Parameter` is not registered in the model) — negligible next to the literal-encoder bias (~0.46).
5. Reasoning step (`time:before` CONSTRUCT) is correct SPARQL but produces all ordered pairs (fully saturated), not
   direct successors as the paper describes; irrelevant to the paper's CARE-SM\*-ts row (no `time:before`).
6. The CARE-SM RGCN runner was not in the repository when the audit started. The original authors shared their CARE-SM
   training notebook (`notebooks/Graphs/rgcn_node_pred_care-sm.ipynb`) on 18 October 2026. It uses the same model as the
   SPHN script, but with embedding size 100, fold seed 42 (for both `StratifiedKFold` and `train_test_split`), 3 layers,
   10 folds, 2,000 epochs and the best validation accuracy checkpoint, and it reads an outcome-free graph from
   `processed_data`. The graph generation notebook still cannot run on `data/syn_data.csv` as committed (it expects an
   `outcome` column; the CSV has `output`).

## 3. Experiments (fold 0, test set of 1,000 patients)

| Run | Graph | Layers | Epochs (best) | AUC | Acc | F1 BH / Rehab / Death |
|---|---|---|---|---|---|---|
| Paper | CARE-SM\*-ts | 3 | 2000 | 0.50 ± 0.02 | 0.44 | 0.53 / 0.30 / 0.00 |
| **A** | CARE-SM-ts, as published | 3 | 400 (36) | **0.494** | 0.447 | 0.61 / 0.11 / 0.00 |
| B | CARE-SM-ts, as published | 5 | stopped at 250, train acc flat 0.433 (chance) | — | — | — |
| E | CARE-SM-ts, literal predicate fixed | 4 | 1000 (996, still rising) | **0.809** | 0.650 | 0.70 / 0.66 / 0.30 |
| F | E + node embeddings N(0,1) (still untrained) | 4 | 600 (441) | **0.882** | 0.743 | 0.83 / 0.68 / 0.62 |
| Paper | SPHN-ts | 3 | 2000 | 0.91 ± 0.01 | 0.78 | 0.83 / 0.76 / 0.66 |
| **S** | SPHN-ts (repo `exp/` generator) | 3 | 1000 (854) | **0.918** | 0.792 | 0.85 / 0.77 / 0.65 |
| G | SPHN-ts with `hasAge` relabelled `hasDiagnosis` | 3 | 600 (599) | **0.678** | 0.570 | 0.63 / 0.59 / 0.00 |
| D | CARE-SM-ts + `hasOutcome` per patient | 3 | 150 (132) | 0.794 | 0.671 | 0.72 / 0.71 / 0.00 |
| D2 | same as D | 3 | 600 (425) | **0.955** | 0.875 | 0.88 / **1.00** / 0.00 |
| — | RF / LR on the CSV, same fold | — | — | 0.914 / 0.918 | 0.788 / 0.789 | |

## 4. Conclusions

- The paper's CARE-SM result (AUC 0.50) reproduces exactly (A: 0.494), but it is an artefact: with 3 layers no feature is
  reachable, numeric literals are dropped by a wrong predicate, and 5 layers on the unfixed pipeline also stays at chance (B).
  Fixing the predicate and using the minimum sufficient depth (4) gives 0.81 (E); making node identities visible gives 0.88 (F).
- The SPHN result (0.91) reproduces (S: 0.918) and equals logistic regression on the same patients (0.918). There is no
  graph-over-tabular advantage on the data the graph pipeline actually uses.
- SPHN beats CARE-SM mainly because age — which alone yields AUC 0.915 — arrives on a dedicated `hasAge` predicate.
  Relabelling that one predicate drops SPHN to 0.678 (G). "Schema compactness" is not the operative factor.
- The extra per-patient relation in the paper's Table 5 is the outcome triple, counted in the statistics only. Both
  generators (`CARESM-graph_generation.ipynb` and `exp/generation/sphn_generation.py`) emit
  `nvasc:synth_patient_N nvasc:hasOutcome nvasc:outcome_X` while the outcome column is present, and drop that column
  before building the graph used for training. Table 5 is one relation and about 10,000 triples (one per patient) above
  the outcome-free graphs for both schemas (SPHN: 15 relations and 1,127,467 triples against 14 and 1,116,191 in the
  committed training graph, which has no outcome node; CARE-SM\*-ts: 13 and 1,754,505 against 12 and 1,744,505), so the
  statistics were computed with the outcome triple in and the models trained without it. With the repo's runner such a
  label edge is exploited (D2: AUC 0.981 over 10 folds), so the CARE-SM 0.50 result is also consistent with an
  outcome-free training graph. The CARE-SM training notebook was not in the repository at the time; the original authors
  shared it on 18 October 2026, and it confirms this (outcome-free graph read from `processed_data`; see the next
  bullet for the results with its own settings). The original authors earlier
  pointed us to the notebook's `nvasc:hasTimePoint` link (October 2026); that link is already among the 12 relations of
  our regenerated graph, which led to this explanation.
- With the notebook's own settings (`run_notebook_caresm.py`, a marked port: embedding 100, fold seed 42, 3 layers,
  2,000 epochs, 10 folds) the published result is reproduced. The graph as published gives AUC 0.502 ± 0.018 over 10
  folds (NB_A), against the paper's 0.50 ± 0.02 and the audit's own run A (0.494 on fold 0). With the literal predicate
  fixed and 4 layers the notebook's settings give 0.822 ± 0.008 (NB_E; audit run E 0.809 on fold 0), and with
  N(0, 1) node embeddings 0.886 ± 0.006 (NB_F; audit run F 0.882 on fold 0). Per-fold files are in
  `results/notebook/`. The notebook does not seed torch, so these runs are not bit-reproducible.
- Training is fragile: with untrained, near-zero node embeddings, even a direct label edge takes ~130–260 epochs to be
  noticed (D/D2 sit at chance until then), so conclusions drawn from a fixed epoch budget can hinge on optimisation.

## Files

- `gen_caresm.py`, `preprocess_caresm.py` — verbatim notebook ports (changes marked `CHANGED`; `--fix_literals` opt-in)
- `run_rgcn.py` — repo RGCN+lit, generalised to L layers; `--emb_std` diagnostic
- `diag_hops.py`, `caresm_roundtrip.py`, `tabular_same_split.py` — diagnostics
- `results/*.json`, `logs/*.log`

## 5. The prediction task itself (data/syn_data.csv)

- 3-class node classification (BackHome 44.3 % / Rehab 43.2 % / Death 12.4 %), transductive, 8,000 / 1,000 / 1,000 patients.
- Not a forecast: inputs include length of stay, number of acts and event timings across the whole stay (known only at discharge).
- The outcome is ~two age thresholds: age ≤ 50.25 → BackHome, ≤ 70.35 → Rehab, else Death gives 10-fold accuracy 0.784
  (RGCN on SPHN: 0.792 here, 0.78 in the paper). Mean age by class 43.0 / 60.1 / 74.8. The Cholesky mixing treats the
  factorised outcome codes 0/1/2 as an ordered scale.
- Care events carry no outcome signal by construction (independent Markov chain; events-only AUC 0.503).
- Out-of-range synthetic values: GCS 11.5–52.0 (mean 17.4; clinical scale 3–15), age 2.9–95.7, stay 0.1–1,764.8 days,
  acts 1.2–505.0; |corr(sex, outcome)| = 0.60.
- CARE-SM runner: no committed code (current tree or git history) trains on a CARE-SM graph; no TransE/RDF2Vec code in the
  current tree. All CARE-SM rows of Table 2 are untraceable to code.
