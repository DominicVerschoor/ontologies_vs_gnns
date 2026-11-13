# Link prediction

Three tail queries `(h, r, ?)`, each a domain-level fact read from the assertions rather than from a graph
predicate. The same question is therefore asked in every graph, including Pure SULO, where team and driver
share one role path and start and finish position share `sulo:hasFeature`.

| Target | Query | Fact | Candidates |
|---|---|---|---|
| Team | `RaceEntry → ?` | `hasTeam` | 182 `Team` entities |
| Position | `RaceEntry → ?` | `hasStartPosition` (grid slot) | 34 `Position` entities |
| Circuit | `Race → ?` | `heldAtCircuit` (Hybrid: `isInCircuit`) | 78 `Circuit` entities |

Every head has exactly one answer, so the filtered and raw rankings are identical. Only nine entity-to-entity
relations exist in all three graphs. The targets are probes of the constructions, each tied to a part of the
schema where the graphs differ:

- **Team** is a participation link, which Pure SULO writes through a Role node and Hybrid and Pure SULO
  restate with `hasTeamMember`.
- **Position** links an entry to a shared hub whose rank Hybrid and Pure SULO reify.
- **Circuit** is the only race-level link.

## Data and protocol (`link_prediction_tools.py`)

- **Graph.** Every fact except `hasFinishPosition`, which is removed for every entry. In Pure SULO it can be
  the same triple as the start position (1,976 entries). Points, status and laps stay.
- **Split.** `race_split`: races 80/10/10 (922 / 115 / 116), `random_state=42`, shared by every graph and both
  encoders. All facts of a race fall into the same set.
- **Hiding.** `hidden_edges` / `visible_graph` remove a held-out fact along its whole rendered path. This
  includes Pure SULO's Role node and its `rdf:type`, found with `f1_population.assertion_lines`. It also
  removes every `hasTeamMember` pair that no visible entry supports. `leaked_facts` checks from the graph that
  no hidden fact is reachable, whether directly, through a Role node, or through
  `entry → Driver → hasTeamMember → Team`.
- **Hidden supervision.** In each training epoch a random 30% of the training races
  (the notebooks' `DISJOINT_TRAIN_RATIO = 0.3`, the value in PyG's `disjoint_train_ratio` tutorial) is hidden and supervised, so the model cannot copy a
  visible edge. Validation hides val + test, and the final test hides test only.
- **Model.** `LinkPredictor` = the node-prediction encoder (`models.RGCNModel` / `GraphSAGEModel`) + DistMult,
  with one vector per target. One model per graph × encoder is trained jointly for all three targets.
- **Loss.** 1-vs-all cross-entropy over all candidates (Dettmers et al. 2018; Ruffinelli et al. 2020), with
  every query weighted equally.
- **Ranking.** A tie counts as the mean of the optimistic and pessimistic rank, so equal scores give no credit (`rank_of_answer`). `ranking_metrics` gives MRR and Hits@1/3/10.
- **Search.** `run_lp_search` uses node prediction's phases and constants, with selection and early stopping
  on pooled val MRR. Dims are capped at 256: R-GCN at 512-d does not fit on Pure SULO. The final model is
  retrained on train + val races and tested with train + val facts as context, over seeds 0–4.
  `target_convergence` reads the per-target validation curves from the saved histories.
- **Ablation.** `run_lp_ablation_variant` follows the node-prediction ablation protocol: the Hybrid winner for
  all 12 variants, and the best validation checkpoint scored on test.
- **Analysis.** `evidence_hops` / `EVIDENCE_CHAINS` give the hop distance from a query to the literals that
  identify its answer, computed from the schemas. `query_ranks_on` and `session_position_entries` re-score
  Position per query.
- **Baselines.** `frequency_baseline` ranks candidates by answer frequency in earlier splits; this is the
  shuffled-answer canary's chance level. `baseline_hits_at_1` gives one-rule baselines:
  - Team: the driver's team that season
  - Circuit: the race name
  - Position: the qualifying position, otherwise the driver's usual grid slot that season

## Notebooks, in run order

| Notebook | What it does | Output (`outputs/f1/link_prediction/`) |
|---|---|---|
| `link_prediction_setup.ipynb` | Sizes and split, same target facts in every graph, leak check at every hiding stage, finish-position removal, baselines | `baselines.csv` |
| `link_prediction_rgcn.ipynb`, `link_prediction_graphsage.ipynb` | E1: shuffled-answer check, full search per graph, final test, comparison with the baselines, per-target convergence, epoch cap | `{rgcn,graphsage}/` |
| `link_prediction_rgcn_ablation.ipynb`, `link_prediction_graphsage_ablation.ipynb` | E2: the 12 ablation graphs, same queries and leak check in every variant, shuffled-answer check on all 12 | `ablation/{rgcn,graphsage}/` |
| `link_prediction_rgcn_depth.ipynb`, `link_prediction_graphsage_depth.ipynb` | E3: all graphs at K = 4 and Pure SULO at K = 3; the GraphSAGE notebook adds a width check (Ad-hoc and Hybrid at 256-d) | `depth/{rgcn,graphsage}/` |
| `link_prediction_analysis.ipynb` | E3, no training: hop table beside the main results, R-GCN Position re-scored per query and split by whether the entry has a session position | `analysis/` |

The ablation, depth and analysis notebooks read the outputs of the main runs.

## Known limitations

These are reported in the thesis and were not changed:

- **Context density.** Training hides about 44% of races per step, tuning evaluation 20% and the final test
  10%, so the test context is denser than the training context.
- **Epoch cap.** Most R-GCN seeds peak near the 400-epoch cap.
- **Position with outcomes visible.** Position is predicted while race outcomes are visible, unlike node
  prediction's outcome mask. This is identical in every graph.
- **Circuit is small.** It is about 2% of the selection signal and has 116 test queries.
- **Duplicated code.** `run_lp_search` / `train_lp_model` duplicate the node-prediction loop, because that
  code is wired to ROC-AUC.

Tests: `tests/test_link_prediction.py`. It covers:

- the facts each graph holds, and hiding along whole paths, including Pure SULO's Role node
- member support and member-route leak detection
- rank limits
- a planted link is learned, and shuffled answers stay at or below the frequency baseline, for both encoders
- per-step hiding, warm-up and early stopping
- out-of-memory handling, and the search phases including resume and the epoch-cap rows
