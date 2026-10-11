# Link Prediction

Link-prediction pipeline: predicting an entity relation, Vehicle/Account/Task, given a case,
evaluated on top of trained R-GCN, GraphSAGE, and TransE models. Every notebook here processes
**one schema (`base` or `sulo`) per kernel session**,  set `SCHEMA_LABEL`/`schema_label` at the
top and re-run for the second schema in a fresh session (or restart python).

## Files

- **`link_prediction_tools.py`**,  shared infrastructure for all three
  `link_prediction_*.ipynb` model notebooks. Includes the induced-subgraph construction
  (`build_induced_subgraph`, BFS-expanded around a sampled set of Case nodes), relation-triple
  extraction and splitting (`extract_direct_triples` / `extract_role_chain_triples`,
  `split_relation_triples`, `split_train_disjoint`), the held-out-edge masking used to keep
  evaluation triples out of the message-passing graph (`mask_held_out_triples`,
  `mask_role_chain_nodes`), type-constrained negative sampling (`build_candidate_pool`,
  `sample_negatives`), the `RGCNEncoder`/`GraphSAGEEncoder`/`DistMultDecoder` model classes, and
  the training/evaluation/hyperparameter-search entry points (`train_link_prediction_model`,
  `evaluate_link_prediction`, `run_small_lp_validation_search`, `run_small_lp_secondary_search`).

- **`link_prediction_setup.ipynb`**,  builds the induced subgraph and extracts the train/val/test
  triples for the three curated link-prediction relations (Vehicle, Account, Task), producing the
  shared data bundle every model notebook below reads.

- **`link_prediction_rgcn.ipynb`** / **`link_prediction_graphsage.ipynb`**,  link prediction on
  top of a trained R-GCN / GraphSAGE encoder (`RGCNEncoder`/`GraphSAGEEncoder` +
  `DistMultDecoder`), structurally parallel to each other.

- **`link_prediction_transe.ipynb`**,  link prediction on top of a trained TransE model (reusing
  the embeddings trained in `node_prediction/transE.ipynb`).

## Running order and outputs

1. **`link_prediction_setup.ipynb`**,  **Outputs:** `{schema}_lp_data.pt` (the shared data bundle
   every model notebook below reads).

2. **`link_prediction_rgcn.ipynb`** / **`link_prediction_graphsage.ipynb`** /
   **`link_prediction_transe.ipynb`**,  **Outputs:** per-model dim-search and (for R-GCN)
   bases-search results CSVs, final tuning results CSV, and the canonical
   `{model}_{schema}_lp_results.csv` per-relation metrics.

## Notes

- Negative sampling is type-constrained (`build_candidate_pool`/`sample_negatives`), not uniform
  over all nodes,  uniform-random global sampling would train against an easier task than the one
  evaluation actually measures.
- Held-out evaluation triples are masked out of the message-passing graph
  (`mask_held_out_triples`/`mask_role_chain_nodes`) rather than only excluded from the loss, so
  they can never leak into the encoder's neighborhood aggregation.
- `link_prediction_transe.ipynb` depends on `node_prediction/transE.ipynb` having already been run
  for the same schema.
