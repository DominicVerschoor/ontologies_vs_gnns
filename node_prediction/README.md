# Node Prediction

Node-classification pipeline: predicting customer-case satisfaction (a binary survey rating) from
the knowledge graph, plus the non-graph tabular baseline used for comparison. Every notebook here
processes **one schema (`base` or `sulo`) per kernel session** — set `SCHEMA_LABEL`/`schema_label`
at the top and re-run for the second schema in a fresh session, unless noted otherwise below.

## Files

- **`gnn_tools.py`** — shared train/eval/ablation infrastructure used by both `RGCN.ipynb` and
  `graph_sage.ipynb`. Includes `train_model`/`evaluate_test_set` (core PyG `NeighborLoader`
  train/eval loops), `run_sequential_optimization_with_verification` (greedy hyperparameter search
  with a verification grid and final held-out test evaluation), `evaluate_final_test_row` /
  `run_final_test_evaluation` (standalone re-run of just the final test evaluation with
  already-tuned hyperparameters), `evaluate_ablation_configuration` / `run_ablation_run` (relation
  ablation studies with class-specific metrics, PR-AUC, and the SNR/RID diagnostics), `run_k_sweep`
  (neighbor-sampling-depth sweep with an OOM retry ladder and resume support), and the
  `get_graph_mask` / `mask_graph_topology` / `compute_entity_reuse_stats` helpers used by the
  ablation studies.

- **`transE.ipynb`** — trains a TransE knowledge-graph embedding model (pykeen) directly on the KG
  triples, with no node features. Used both as a node-classification baseline (a downstream
  classifier trained on the resulting Case embeddings) and reused by `link_prediction/`.

- **`RGCN.ipynb`** — main R-GCN node-classification pipeline: builds the PyG graph from the `.nt`
  file and BERT embeddings, then runs hyperparameter tuning, feature/relation ablation, and a
  K-hop depth sweep.

- **`graph_sage.ipynb`** — main GraphSAGE node-classification pipeline, structurally parallel to
  `RGCN.ipynb` (same `model_factory`/`base_model_factory`/`sulo_model_factory`/
  `k_sweep_model_factory` pattern) so the two stay easy to compare and any shared-behavior change
  should go in `gnn_tools.py` rather than being duplicated in both notebooks.

- **`RGCN_k_sweep.ipynb`** — standalone K-hop depth sweep containing both schemas in one notebook
  (unlike every other notebook's one-schema-per-run convention), for isolated re-runs without
  repeating full hyperparameter tuning.

- **`automl_tabular_baseline.ipynb`** — independent of the KG; builds a flat, one-row-per-case
  feature table directly from the raw source tables (case/account/vehicle fields, aggregated
  Task/CaseHistory counts, raw text embeddings) and trains XGBoost, using the same case split as
  `transE.ipynb` for a fair comparison.

## Running order and outputs

1. **`transE.ipynb`**: **Outputs:** trained entity embeddings (`*_final_transe_embeddings.pt`),
   tuning results CSV, the canonical case-level train/val/test split (`transe/{schema}_split.pt`)
   reused by `automl_tabular_baseline.ipynb`, masked training triples files
   (`merged_{schema}_training.nt`).

2. **`RGCN.ipynb`** / **`graph_sage.ipynb`**: **Outputs:** `{schema}_vars.pt` (graph tensors +
   splits + vector table), `{schema}_ontology_enums.pt`, hyperparameter tuning results CSV
   (`{rgcn,graphsage}/{schema}_sequential_tuning_results.csv`), ablation study CSV
   (`{schema}_ablation_study_results.csv`), K-sweep CSV (`{schema}_k_sweep_results.csv`), trained
   model checkpoints.

3. **`RGCN_k_sweep.ipynb`** (optional, isolated re-run) — **Outputs:** K-sweep results CSV per
   schema, same format as `RGCN.ipynb`'s own K-sweep section.

4. **`automl_tabular_baseline.ipynb`**: **Outputs:** `ds_ccc.kg_automl_baseline_table`,
   `tabular_baseline_results.csv`.

## Notes

- **`num_workers=0` is required** on every PyG `NeighborLoader` in `gnn_tools.py` — set
  deliberately to fix a host-RAM SIGKILL on single-node clusters and should not be reverted
  without a reason.
- Call `cleanup()` (`gc.collect()` + `torch.cuda.empty_cache()`) after every model train/eval
  cycle inside a loop (`run_k_sweep`, `run_sequential_optimization_with_verification`) — GPU
  memory is not freed automatically between iterations. Models must be moved `.cpu()` and
  `del`eted before `cleanup()` for the cache to actually release.
