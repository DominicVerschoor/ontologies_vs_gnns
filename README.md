# Ontologies vs. GNNs: upper-level ontology alignment and graph learning on Formula 1 data

Thesis code. The same Formula 1 records are built into **three knowledge graphs** that differ in how far
they follow the Simplified Upper-Level Ontology ([SULO](https://w3id.org/sulo/)). Each graph is then
evaluated with two GNN encoders, **R-GCN** and **GraphSAGE** (mean aggregator), on **node classification**
and **link prediction**. Every graph gets the same records, splits, seeds, tuning budget and evaluation
code, so score differences can be attributed to the graph construction.

Data: the public Kaggle dataset
[Formula 1 World Championship (1950–present)](https://www.kaggle.com/datasets/atharvranjan/formula-1-world-championship-1950-present),
pinned to version 9 and downloaded with `kagglehub`. No credentials are needed.

## The three knowledge graphs

| Graph | Schema | Construction |
|---|---|---|
| **Ad-hoc** | `turtles/f1/base_kg.ttl` | Hand-made F1 domain schema: domain-specific properties, attributes as direct literals. No SULO. |
| **Hybrid** | `turtles/f1/hybrid_sulo_kg.ttl` | Ad-hoc's domain properties, every class aligned under a SULO class, plus the SULO relations that the alignment makes available: `precedesRace` / `precedesSeason` (`sulo:precedes`), `hasTeamMember` (`sulo:hasItem`), and the grid rank on an information object (`sulo:isReferredToIn` → `sulo:hasValue`). `heldAtCircuit` is renamed `isInCircuit` (`sulo:isIn`). |
| **Pure SULO** | `turtles/f1/pure_sulo_kg.ttl` | The same classes, alignments and enrichments as Hybrid, but every property is written with SULO predicates only. Participation goes through a `sulo:Role` node (PRO pattern), and literals go through an intermediate information object, quantity or time node (SOLID pattern). |

Domain classes: `Season`, `Race`, `Circuit`, `RaceEntry`, `SessionResult`, `Driver`, `Team`, `Position`.
`turtles/sulo.ttl` is the vendored upstream SULO ontology. The three `.ttl` schemas were generated with the
SULO-Compliant Schema Builder.

The tables are turned once into graph-independent **assertions** (`f1_population.build_assertions`). Each
schema then decides how an assertion is written as triples (`read_schema` parses the `rdfs:domain` class
expressions, `render_ntriples` writes them). No triple shape is written by hand, so the graphs hold the same
base facts by construction. Hybrid and Pure SULO hold the same facts and differ only in structure.

## Tasks

- **Node classification**: did a `RaceEntry` score championship points (points > 0, 31.3% positive)?
  The label and the race outcomes (finish position, status, laps completed) are removed for every entry.
  Stratified 80/10/10 split over the 27,032 entries. Headline metric: ROC-AUC.
- **Link prediction**: three tail queries, each ranked against every entity of the answer class:
  - Team of a `RaceEntry` (182 candidates)
  - start (grid) Position of a `RaceEntry` (34 candidates)
  - Circuit of a `Race` (78 candidates)

  Holdout is per race (80/10/10 of 1,153 races). A held-out fact is hidden along its whole rendered path,
  and `leaked_facts` checks that nothing hidden is reachable. Metrics: MRR and Hits@1/3/10, per target and
  pooled.

## Experiments → code

| Experiment | Question | Notebooks | Results in `outputs/f1/` |
|---|---|---|---|
| E1: main comparison (each graph tuned) | Ad-hoc vs. Hybrid vs. Pure SULO | `node_prediction/RGCN.ipynb`, `graph_sage.ipynb`; `link_prediction/link_prediction_{rgcn,graphsage}.ipynb` | `node_prediction/{rgcn,graphsage}/`, `link_prediction/{rgcn,graphsage}/` |
| E2: enrichment ablation (one unit at a time, both directions) | Effect of each SULO enrichment | `node_prediction/{RGCN,graph_sage}_ablation.ipynb`; `link_prediction/link_prediction_{rgcn,graphsage}_ablation.ipynb` | `*/ablation/{rgcn,graphsage}/` |
| E3: depth | Is the effect explained by the hop distance to the evidence? | `node_prediction/{RGCN,graph_sage}_k_sweep.ipynb`, `RGCN_ablation_k2.ipynb`; `link_prediction/link_prediction_{rgcn,graphsage}_depth.ipynb`, `link_prediction_analysis.ipynb` | `node_prediction/k_sweep/`, `node_prediction/ablation_k2/`, `link_prediction/depth/`, `link_prediction/analysis/` |

Each notebook states the literature source of every hyperparameter in its constants cell. Predictions
for the follow-up checks are written in the notebook's introduction, before its results.

## Repository layout

```
f1_population.py             Kaggle tables -> assertions -> triples per schema (scope, IRIs, schema parsing, rendering)
f1_graph_population.ipynb    Runs the population; writes data/f1/ (see below)
f1_node_data.py              Shared graph tensors: masking, splits, node features, ablation schemas (build_kg_data)
ontology_visualization.ipynb Schema diagrams of the three .ttl files
results_figures.py           Figures for the Results chapter, read from outputs/f1/ (no training)

node_prediction/             Node classification: models, training/tuning code, notebooks (see its README)
link_prediction/             Link prediction: data, model, training/tuning code, notebooks (see its README)
turtles/f1/                  The three F1 schemas;  turtles/sulo.ttl: upstream SULO
tests/                       pytest suite (population, graph data, models, search, link prediction)
outputs/f1/                  Result CSVs and figures (model checkpoints *.pt are not versioned)
```

`data/f1/` is generated and not versioned. It holds `assertions.csv` and `entities.csv`, the complete graphs
`{adhoc,pure_sulo,hybrid_sulo}.nt`, and `sample/<kg>.ttl` (the 20 most recent races, for reading). Reload the
CSVs with `dtype=str, keep_default_na=False`, otherwise lexical forms change (`true` → bool, `78.050` → `78.05`).

## Setup and running

```bash
conda env create -f environment.yml      # env "sulo": Python 3.11, torch 2.7 + CUDA 12.6, PyG; Graphviz for the diagrams
conda activate sulo
pytest                                    # ~260 tests, about 30 s
```

1. `f1_graph_population.ipynb`: downloads the data and writes `data/f1/`.
2. `node_prediction/`: main runs first (the ablation and K-sweep read the tuning CSVs). See `node_prediction/README.md`.
3. `link_prediction/`: setup checks, then main runs, ablation, depth and analysis. See `link_prediction/README.md`.
4. `python results_figures.py`: writes the figures to `outputs/f1/figures/`.

Notebooks can be run from the repo root or from their own folder. The runs were done on one RTX 3060 (12 GB);
the largest configuration peaks at about 9.6 GiB. Every notebook first calls
`gnn_tools.limit_cuda_memory_to_free_vram()`, so an over-large configuration raises an out-of-memory error
that is logged and skipped. (Under WSL2 it would otherwise spill into system RAM.)

**Reproducibility.** The splits use `random_state=42`. Tuning uses seeds 0–2, and final results are the
mean ± std over seeds 0–4. CPU runs are bit-reproducible. On the GPU, PyG's scatter operations are not
deterministic, so repeated runs differ slightly. This noise is measured and reported as a limitation.
Long sweeps write their CSVs incrementally and resume after a crash.

## Legacy files

These files come from an earlier phase of the project, on a different, non-public dataset. They are not
used by the F1 pipeline and are out of scope for the thesis:

- `Graph_population.ipynb`, `embeddings.ipynb`, `analysis.ipynb`, `extract_graph_geometry.ipynb`
- `training_data.py`
- `turtles/cac/`
- `node_prediction/transE.ipynb`, `node_prediction/automl_tabular_baseline.ipynb`
- `link_prediction/link_prediction_transe.ipynb`
- the concept-masking helpers in `node_prediction/gnn_tools.py` (`get_graph_mask`, `mask_graph_topology`,
  `compute_entity_reuse_stats`)

TransE and the tabular baseline are not part of the evaluation.
