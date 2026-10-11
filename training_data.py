import re
import torch
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split


class Ontology:
    """
    Parses an .nt file into PyG-style tensors (edge_index, edge_type) plus
    node/relation lookup dicts, and separately extracts the Case-level survey
    rating as a label dict -- excluding those specific edges from the graph
    topology so the label can never leak into message passing.
    """

    def __init__(
        self,
        nt_file_path: str,
        label_predicate: str,
        label_object_pattern: str = r"SurveyRating/(\d+)",
    ):
        """
        nt_file_path: path to the .nt data file (instance-level triples --
            NOT the .ttl ontology/schema file).

        label_predicate: substring identifying the predicate that links
            Case -> SurveyRating in this ontology variant.
                - sulo_kg-derived data: "hasFeature"
                  (ambiguous alone -- disambiguated via label_object_pattern)
                - base-derived data: "hasSurveyRating"
                  (unambiguous alone)

        label_object_pattern: regex applied to the OBJECT uri of any edge
            whose predicate matches `label_predicate`. A match confirms the
            edge is really the SurveyRating edge (not some other
            hasFeature-typed attribute) and captures the binary rating in
            group(1).
        """
        self.enum_nodes = {}
        self.enum_relations = {}
        self.survey_labels = {}  # case_uri -> float rating (0.0 / 1.0)
        self._label_pattern = re.compile(label_object_pattern)
        self.label_predicate = label_predicate

        src, dst, rel = [], [], []

        def node_id(uri):
            if uri not in self.enum_nodes:
                self.enum_nodes[uri] = len(self.enum_nodes)
            return self.enum_nodes[uri]

        def relation_id(uri):
            if uri not in self.enum_relations:
                self.enum_relations[uri] = len(self.enum_relations)
            return self.enum_relations[uri]

        print(f"Parsing N-Triples from {nt_file_path}...")
        skipped_malformed = 0
        with open(nt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", maxsplit=2)
                if len(parts) < 3:
                    skipped_malformed += 1
                    continue

                subject = parts[0]
                predicate = parts[1]
                obj = parts[2].rstrip(" .")

                s_id = node_id(subject)

                # Disambiguate + isolate the survey-rating edge. If it matches,
                # record the label and skip adding it to the topology -- this
                # is the leakage guard.
                if label_predicate in predicate:
                    match = self._label_pattern.search(obj)
                    if match:
                        self.survey_labels[subject] = float(match.group(1))
                        continue

                o_id = node_id(obj)
                r_id = relation_id(predicate)

                # Forward + inverse edge, doubled relation indexing (rel*2 /
                # rel*2+1), consistent with the RotatE training convention.
                src.append(s_id)
                dst.append(o_id)
                rel.append(r_id * 2)

                src.append(o_id)
                dst.append(s_id)
                rel.append(r_id * 2 + 1)

        if skipped_malformed:
            print(f"Skipped {skipped_malformed} malformed/short lines.")

        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.edge_type = torch.tensor(rel, dtype=torch.long)
        self.num_nodes = len(self.enum_nodes)

        print(
            f"Parsed {self.num_nodes} nodes, {len(self.enum_relations)} base "
            f"{self.edge_index.shape[1]} directed edges."
        )
        print(
            f"Found {len(self.survey_labels)} case-level survey ratings "
        )
        if len(self.survey_labels) == 0:
            print(
                "WARNING: zero labels found. Check that label_predicate and "
                "label_object_pattern actually match this .nt file's URI "
                "conventions -- run `grep SurveyRating yourfile.nt | head` "
                "to confirm the format."
            )

    def build_feature_lookup(self, emb_dim: int, embedding_df: pd.DataFrame, task_embedding_df: pd.DataFrame,
                            task_description_predicate: str = "hasTaskDescription",
                            disable_text: bool = False, disable_categorical: bool = False):
        node_dict = self.enum_nodes

        task_bert_dict = dict(zip(task_embedding_df['task_id'].astype(str), task_embedding_df['task_desc_vector']))
        case_desc_dict = dict(zip(embedding_df['case_id'].astype(str), embedding_df['description_vector']))
        case_sol_dict = dict(zip(embedding_df['case_id'].astype(str), embedding_df['solution_vector']))

        def escape_literal(text):
            return str(text).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')

        # Case description/solution -- embedding_df has raw text, so literal reconstruction + match still works
        baseline_text_dict = {}
        if 'case_description' in embedding_df.columns and 'case_solution_final_statement' in embedding_df.columns:
            for _, row in embedding_df.iterrows():
                desc_str = f"\"{escape_literal(row['case_description'])}\"^^<http://www.w3.org/2001/XMLSchema#string>"
                sol_str = f"\"{escape_literal(row['case_solution_final_statement'])}\"^^<http://www.w3.org/2001/XMLSchema#string>"
                if isinstance(row['description_vector'], (list, np.ndarray)):
                    baseline_text_dict[desc_str] = row['description_vector']
                if isinstance(row['solution_vector'], (list, np.ndarray)):
                    baseline_text_dict[sol_str] = row['solution_vector']

        # hasTaskDescription edge directly instead of reconstructing/matching its text.
        task_literal_vectors = {}  # literal node index -> vector, populated only if this predicate exists
        matching_rels = [uri for uri in self.enum_relations if task_description_predicate in uri]
        if matching_rels:
            if len(matching_rels) > 1:
                print(f"Warning: multiple relations match '{task_description_predicate}': {matching_rels}. Using first.")
            rel_id = self.enum_relations[matching_rels[0]]
            forward_rel_id = rel_id * 2  # matches the forward-edge convention from parsing

            mask = self.edge_type == forward_rel_id
            src_nodes = self.edge_index[0][mask].tolist()
            dst_nodes = self.edge_index[1][mask].tolist()

            # Reverse-lookup only for the (small) set of Task nodes actually on this predicate,
            # not the whole graph -- avoids building a full num_nodes-sized reverse dict.
            needed_src = set(src_nodes)
            idx_to_uri = {idx: uri for uri, idx in node_dict.items() if idx in needed_src}

            linked = 0
            for s, d in zip(src_nodes, dst_nodes):
                task_uri = idx_to_uri.get(s, "").strip('<>')
                if '/Task/' not in task_uri:
                    continue
                task_id = task_uri.split('/')[-1]
                if task_id in task_bert_dict:
                    task_literal_vectors[d] = torch.tensor(task_bert_dict[task_id], dtype=torch.float)
                    linked += 1
            print(f"Linked {linked} task descriptions via '{matching_rels[0]}' edge traversal.")
        else:
            print(f"No relation matching '{task_description_predicate}' found -- "
                f"skipping task-description linking for this graph.")

        vector_table = []
        key_to_row = {}

        def deterministic_vector(key: str) -> torch.Tensor:
            seed = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**31)
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(emb_dim, generator=gen)

        def get_row(key: str, vector=None) -> int:
            if key not in key_to_row:
                key_to_row[key] = len(vector_table)
                vector_table.append(vector if vector is not None else deterministic_vector(key))
            return key_to_row[key]

        node_index = torch.zeros(self.num_nodes, dtype=torch.long)
        matched_text, fallback_count = 0, 0

        for node_uri, idx in node_dict.items():
            uri_string = str(node_uri).strip('<>')
            is_literal = uri_string.startswith('"')

            if '/' in uri_string and not is_literal:
                semantic_marker = uri_string.split('/')[-2]
                entity_id = uri_string.split('/')[-1]
            else:
                semantic_marker = uri_string
                entity_id = uri_string

            row = None

            if not disable_text:
                if semantic_marker == 'InformationObject':
                    if entity_id.endswith('_case_description'):
                        case_id = entity_id.replace('_case_description', '')
                        if case_id in case_desc_dict and isinstance(case_desc_dict[case_id], (list, np.ndarray)):
                            row = get_row(f"case_desc:{case_id}", torch.tensor(case_desc_dict[case_id], dtype=torch.float))
                            matched_text += 1
                    elif entity_id.endswith('_case_solution_final_statement'):
                        case_id = entity_id.replace('_case_solution_final_statement', '')
                        if case_id in case_sol_dict and isinstance(case_sol_dict[case_id], (list, np.ndarray)):
                            row = get_row(f"case_sol:{case_id}", torch.tensor(case_sol_dict[case_id], dtype=torch.float))
                            matched_text += 1
                    elif entity_id.endswith('_task_description'):
                        task_id = entity_id.replace('_task_description', '')
                        if task_id in task_bert_dict:
                            row = get_row(f"task_desc:{task_id}", torch.tensor(task_bert_dict[task_id], dtype=torch.float))
                            matched_text += 1

                elif is_literal:
                    if uri_string in baseline_text_dict:
                        row = get_row(f"lit:{uri_string}", torch.tensor(baseline_text_dict[uri_string], dtype=torch.float))
                        matched_text += 1
                    elif idx in task_literal_vectors:
                        row = get_row(f"tasklit:{idx}", task_literal_vectors[idx])
                        matched_text += 1

            if row is None and not disable_categorical:
                categorical_key = semantic_marker if not is_literal else entity_id
                row = get_row(categorical_key)
                fallback_count += 1
            elif row is None:
                row = get_row("__zero__", torch.zeros(emb_dim, dtype=torch.float))

            node_index[idx] = row

        vector_table = torch.stack(vector_table)
        print(f"Vector table: {vector_table.shape[0]} unique rows "
            f"({matched_text} real text-backed node instances, {fallback_count} categorical fallbacks)")

        return vector_table, node_index


class TrainingData:
    """
    Builds train (80%) / val (10%) / test (10%) node-id splits from
    `sulo_graph.survey_labels`. The label already sits on the Case node
    directly (resolved during SuloKnowledgeGraph parsing) -- no traversal
    needed here, just a split.
    """

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def create_training_data(self, sulo_graph: Ontology) -> None:
        if not sulo_graph.survey_labels:
            print(
                "Error: sulo_graph has no survey labels. Nothing to split -- "
                "check the parsing warnings above."
            )
            return

        labeled_nodes = []
        labels = []

        for case_uri, score in sulo_graph.survey_labels.items():
            case_node_id = sulo_graph.enum_nodes.get(case_uri)
            if case_node_id is None:
                print(f"Warning: label subject not found among parsed nodes: {case_uri}")
                continue
            labeled_nodes.append(case_node_id)
            labels.append(score)

        print(f"Mapped {len(labeled_nodes)} survey scores directly to Case nodes.")

        x_train, x_temp, y_train, y_temp = train_test_split(
            labeled_nodes, labels, test_size=0.2, random_state=42, shuffle=True
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
        )

        self.x_train = torch.tensor(x_train, dtype=torch.long)
        self.y_train = torch.tensor(y_train, dtype=torch.float)
        self.x_val = torch.tensor(x_val, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.float)
        self.x_test = torch.tensor(x_test, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.float)