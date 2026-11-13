# Node prediction

Binary classification of `RaceEntry` nodes: did the entry score championship points (`hasRacePoints`,
points > 0, 31.3% positive)? The same code runs on all three graphs (`adhoc`, `hybrid_sulo`, `pure_sulo`).

## Data (`f1_node_data.py`, repo root)

- **Masking.** `masked_lines` removes the label and the race outcomes (`hasFinishPosition`, `hasStatus`,
  `hasLapsCompleted`) from the assertions **before** rendering, for every entry, not only test entries. The
  finish position alone gives ROC-AUC 0.985, and a test entry's finish position would otherwise follow from
  its race-mates'. Because masking happens before rendering, a Pure SULO path disappears together with its
  intermediate node. Facts known before the race stay: grid, qualifying, sprint and practice results.
- **Labels** come from the assertions (`race_entry_labels`), never from a graph.
- **Split.** `stratified_split`: 80/10/10, stratified by label, `random_state=42`, computed once on IRIs and
  shared by every graph and both models (21,625 / 2,703 / 2,704). A temporal split was not used, because the
  label base rate and the available session data change by era.
- **Graph.** `build_graph` adds an inverse edge for every triple, which R-GCN gets as its own relation type
  (Schlichtkrull et al. 2018). `rdf:type` edges stay.
- **Features.** `node_features` never reads IRI text:
  - an instance is described by an embedding of its `rdf:type` set
  - a class node by its class
  - a numeric, boolean or date literal by its datatype embedding × its standardised value
  - a string literal by one learnable embedding per distinct string

  A degree feature is added.
- **Ablation schemas.** `ABLATION_UNITS` lists the five enrichment units (`sulo_typing`, `precedesRace`,
  `precedesSeason`, `hasTeamMember`, `rank_reification`). `ablation_schema` / `swap_units` build the 12
  `ABLATION_VARIANTS`: `hybrid_minus_<unit>`, `adhoc_plus_<unit>`, and the two controls. A variant is a
  schema, so masking and splits go through the normal path.

## Code

- **`models.py`**
  - `RGCNModel` and `GraphSAGEModel` share `NodeEncoder` and the same `forward`, so the training loop never
    branches on the architecture.
  - R-GCN uses basis decomposition and runs each layer under gradient checkpointing, without which Pure SULO
    does not fit in memory at 256-d × 3. Logits and gradients are unchanged (tested).
  - GraphSAGE is GraphSAGE-mean (`aggr='mean'`, `project=False`), as in the original `MeanAggregator`.
- **`gnn_tools.py`**
  - **Training.** `train_model` trains full-batch: each epoch is one forward pass over the whole graph and
    one Adam step. Early stopping uses validation ROC-AUC with patience 50 and a 50-epoch warm-up
    (`start_from_epoch`), and the best weights are restored. `train_final_model` retrains on train + val,
    and `evaluate_test_set` scores the test set.
  - **Search.** `run_sequential_optimization_with_verification` runs four phases. Every configuration runs
    on tuning seeds 0–2 and is scored by mean val ROC-AUC (`run_config_over_seeds`).
    1. A joint hidden-dim × depth grid.
    2. Greedy R-GCN bases → learning rate → weight decay.
    3. A 2×2 verification grid.
    4. The winner is retrained on train + val for its median best epoch + 1 and tested once per final seed
       0–4 (`evaluate_final_test_row`).

    Failed and out-of-memory configurations are logged, not fatal. `finished_tuning_results` lets a notebook
    resume after a crash.
  - **Ablation.** `run_ablation_variant` / `summarize_ablation`: every variant uses the tuned Hybrid
    configuration. Per seed, it early-stops on validation and scores the best checkpoint on test.
  - **K-sweep.** `tuned_winner` / `run_k_sweep` / `summarize_k_sweep`: each graph keeps its own tuned
    configuration and only the depth K ∈ {1, 2, 3, 4} changes. The tuned K = 3 rows are reused.
  - **Checks.** `runs_ended_by_epoch_cap` flags seeds whose best epoch is within `PATIENCE` of the 400-epoch
    cap.

## Notebooks, in run order

| Notebook | What it does | Output (`outputs/f1/node_prediction/`) |
|---|---|---|
| `RGCN.ipynb`, `graph_sage.ipynb` | E1: shuffled-label check on the real graphs, full search per graph, final test over 5 seeds | `{rgcn,graphsage}/<kg>_sequential_tuning_results.csv` |
| `RGCN_ablation.ipynb`, `graph_sage_ablation.ipynb` | E2: the 12 ablation graphs at the Hybrid winner, shuffled-label check on all 12 | `ablation/{rgcn,graphsage}/` |
| `RGCN_k_sweep.ipynb`, `graph_sage_k_sweep.ipynb` | E3: K = 1–4 per graph at its own winner | `k_sweep/{rgcn,graphsage}/k_sweep_results.csv`, `k_sweep_summary.csv`, plot |
| `RGCN_ablation_k2.ipynb` | E3: controls and `rank_reification` variants at K = 2 (R-GCN) | `ablation_k2/rgcn/` |

The ablation and K-sweep notebooks read the tuning CSVs, so the main notebooks run first. `RGCN.ipynb`
and `graph_sage.ipynb` are generated from one template and differ only in the model. The same holds for
each ablation pair and each K-sweep pair. Shared logic lives in the `.py` modules.

Earlier runs are kept for reference and are not part of the reported results:

- `rgcn_single_seed/`: the first run, with one tuning seed.
- `rgcn_sampled/`: neighbour sampling instead of full-batch training. Sampling almost never drew a
  `Position` hub's single rank edge, so the model could not see the grid position.
- `{rgcn,graphsage}_no_warmup/`: tuning without the early-stopping warm-up. Seeds kept validation spikes
  from the first epochs as their best.

## Notes

- The wall-clock times of R-GCN and GraphSAGE are not comparable, because only R-GCN checkpoints its
  layers.
- In loops, move every model `.cpu()`, `del` it, and call `cleanup()` before the next iteration, or GPU
  memory is not released.
- Tests: `tests/test_f1_node_data.py`, `tests/test_models.py` and `tests/test_gnn_tools.py`. They cover:
  - leakage: a random-label canary, and flipping val/test labels leaves training bit-identical
  - a hub-identity regression: R-GCN must read a hub's single rank edge among 1,250 neighbours
  - seeded reproducibility and checkpointing equivalence
  - the search: phases, seed aggregation, reuse, out-of-memory handling, the epoch-cap rule, resume
