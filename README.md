# Knowledge Graph Satisfaction Prediction

Research pipeline comparing knowledge-graph-based and non-graph approaches on two tasks over a
customer-case knowledge graph: **node classification** (predicting customer-case satisfaction, a
binary survey rating) and **link prediction** (predicting an entity relation, Vehicle/Account/Task, given a case). Both tasks are evaluated across two knowledge graph constructions, a formalized Upper Level Ontology (ULO) based
ontology (**SULO**) and an ad-hoc schema. They were tested over three learning model: R-GCN, GraphSAGE, and TransE,
alongside a non-graph tabular baseline (XGBoost) for the node-classification task.

## Repository layout

```
Graph_population.ipynb       Populates the KG (RDF triples) from raw source tables -> .nt/.ttl files
embeddings.ipynb             Generates BERT text embeddings (case/task descriptions) -> Delta tables
extract_graph_geometry.ipynb Structural comparison of the SULO vs. Ad-hoc KGs (Table 1)
ontology_visualization.ipynb Renders the SULO/Ad-hoc class-and-property diagrams
analysis.ipynb               Aggregates results from every pipeline stage into the final comparison tables
training_data.py             Ontology/TrainingData: parses .nt into PyG tensors + the 80/10/10 split

node_prediction/     Node-classification pipeline: TransE, R-GCN, GraphSAGE, and the
                     XGBoost tabular baseline, plus their shared gnn_tools.py infrastructure.
                     See node_prediction/README.md.
link_prediction/     Link-prediction pipeline: R-GCN/GraphSAGE/TransE-based prediction of the
                     Vehicle/Account/Task relations, plus their shared link_prediction_tools.py
                     infrastructure. See link_prediction/README.md.

turtles/             sulo.ttl (vendored upstream SULO ontology) plus this project's own schema
                     extension (base_kg.ttl, sulo_kg.ttl). If you wish to build a new knowledge graph, you will need to create your own ttl, and modify Graph_population.ipynb to fit your data
```

## Running the pipeline

Run in this order. Most notebooks process **one schema (`base` or `sulo`) per kernel session (or restart python)**,
set `SCHEMA_LABEL`/`schema_label` at the top and re-run for the second schema in a fresh session.

1. **`Graph_population.ipynb`**, builds RDF triples for both KG schemas from the raw source
   tables (Case/Account/Vehicle/Task/CaseHistory/Survey), scoped to *all* cases with a survey
   response, not just the balanced training subset.
   **Outputs:** `.nt` instance files per schema (e.g.
   `subset_balanced_baseline_populated_graph.nt`, `subgraph_balanced_sulo_populated_graph.nt`),
   plus a full/unbalanced variant of each; materializes `ds_ccc.kg_sulo_balanced_case_ids` (the
   stratified 50k-case sample every downstream training notebook subsets to).

2. **`embeddings.ipynb`**, generates BERT embeddings for case descriptions, case solutions, and
   task descriptions, over the same full case scope as step 1.
   **Outputs:** `ds_ccc.sulokg_embeddings_dv` (case description/solution vectors),
   `ds_ccc.sulokg_task_embeddings_dv` (task description vectors).

3. **`node_prediction/`**, the node-classification pipeline: TransE (also reused as an
   embedding source by `link_prediction/`), R-GCN, GraphSAGE, the standalone K-sweep notebook, and
   the XGBoost tabular baseline. See `node_prediction/README.md` for run order and per-notebook
   outputs.

4. **`extract_graph_geometry.ipynb`**, measures structural differences between the SULO and
   Ad-hoc KGs (node/edge counts, degree distribution, feature-to-Case proximity).
   **Outputs:** `rq1a_structural_comparison.csv` (Table 1), `{schema}_raw_edges.pt` (uncapped
   graph, cached separately from `node_prediction/`'s degree-capped training graph).

5. **`link_prediction/`**, the link-prediction pipeline: builds a shared induced-subgraph data
   bundle, then runs link prediction on top of trained R-GCN, GraphSAGE, and TransE models. See
   `link_prediction/README.md` for run order and per-notebook outputs.

6. **`analysis.ipynb`**, reads only the CSV/artifact outputs of the steps above; produces the
   final aggregated comparison tables and figures. Nothing else depends on its output.

## Results

Final comparison tables/figures are produced by `analysis.ipynb` from each pipeline stage's own
output CSVs, not hardcoded here, see that notebook for current numbers.
