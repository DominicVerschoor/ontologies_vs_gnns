"""
Turns the populated F1 KGs into node-prediction inputs, identically for all three KGs:
label extraction, leakage masking, a shared stratified split, graph tensors and node
features.

The label and every race-outcome property are dropped from the assertions *before* each
KG is rendered (f1_population.render_ntriples), so a masked fact never exists in any
graph shape -- including Pure SULO's intermediate-node paths, and Pure's merged
start/finish position triple, where only the start position survives.

Node features never read IRI text: an IRI node is described by its rdf:type set (or, for
a class node, by being that class), a literal by its datatype and value.
"""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from f1_population import RDF_TYPE, SCHEMA_FILES, XSD_NS, Schema, read_schema, render_ntriples

LABEL_PROPERTY = "hasRacePoints"
# Known only after the race; hasFinishPosition alone predicts the label with ROC-AUC 0.985.
OUTCOME_PROPERTIES = frozenset({"hasFinishPosition", "hasStatus", "hasLapsCompleted"})

# What Hybrid adds to Ad-hoc, one unit per piece of SULO semantics, so each can be ablated alone.
# sulo_typing owns Schema.sulo_superclass (instances also typed with their SULO superclass); the
# other units own schema properties. heldAtCircuit -> isInCircuit is not a unit: it renames the
# predicate and leaves the edges unchanged.
ABLATION_UNITS = {
    "sulo_typing": frozenset(),
    "precedesRace": frozenset({"precedesRace"}),
    "precedesSeason": frozenset({"precedesSeason"}),
    "hasTeamMember": frozenset({"hasTeamMember"}),
    "rank_reification": frozenset({"hasRank"}),   # Position/<n>/hasRank -> sulo:hasValue vs a direct literal
}
# variant -> (base KG, donor KG, units whose shape comes from the donor). The base KG is the
# variant's control: leave-one-out from Hybrid, add-one-in to Ad-hoc.
ABLATION_VARIANTS = {
    "hybrid_sulo": ("hybrid_sulo", "adhoc", frozenset()),
    "adhoc": ("adhoc", "hybrid_sulo", frozenset()),
    **{f"hybrid_minus_{u}": ("hybrid_sulo", "adhoc", frozenset({u})) for u in ABLATION_UNITS},
    **{f"adhoc_plus_{u}": ("adhoc", "hybrid_sulo", frozenset({u})) for u in ABLATION_UNITS},
}

NUMERIC_DATATYPES = {XSD_NS + t for t in ("integer", "nonNegativeInteger", "decimal")}
_LITERAL = re.compile(r'^"(.*)"\^\^<([^>]+)>$')


def load_population(data_dir="data/f1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entities, assertions) as written by f1_graph_population.ipynb, lexical forms unchanged."""
    read = lambda name: pd.read_csv(Path(data_dir) / name, dtype=str, keep_default_na=False)
    return read("entities.csv"), read("assertions.csv")


def race_entry_labels(assertions: pd.DataFrame) -> pd.Series:
    """RaceEntry IRI -> 1 if it scored points, else 0. Read from the assertions, never from a graph."""
    rows = assertions[assertions["property"] == LABEL_PROPERTY]
    assert set(rows["value"]) <= {"true", "false"}, "hasRacePoints must be an xsd:boolean lexical form"
    return rows.set_index("subject")["value"].eq("true").astype(int).sort_index()


def masked_lines(entities: pd.DataFrame, assertions: pd.DataFrame, schema) -> list[str]:
    """The KG rendered without the label and the race outcomes."""
    kept = assertions[~assertions["property"].isin(OUTCOME_PROPERTIES | {LABEL_PROPERTY})]
    lines, _ = render_ntriples(entities, kept, schema)
    return lines


def stratified_split(labels: pd.Series, seed=42) -> tuple[list[str], list[str], list[str]]:
    """80/10/10 split of the labelled IRIs, stratified by label. Shared by all KGs and models."""
    iris, y = labels.index.tolist(), labels.tolist()
    train, rest, _, y_rest = train_test_split(iris, y, test_size=0.2, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return train, val, test


def swap_units(base: Schema, donor: Schema, units) -> Schema:
    """
    base with each unit's shape taken from donor. A property the donor lacks is dropped, so
    render_ntriples sets its assertions aside and the facts vanish from the graph.
    """
    properties = dict(base.properties)
    for prop in set().union(*(ABLATION_UNITS[u] for u in units)):
        properties.pop(prop, None)
        if prop in donor.properties:
            properties[prop] = donor.properties[prop]
    superclass = donor.sulo_superclass if "sulo_typing" in units else base.sulo_superclass
    return Schema(properties, dict(superclass))


def ablation_schema(variant: str, schemas: dict[str, Schema]) -> Schema:
    """The schema of an ABLATION_VARIANTS entry, from the parsed {kg: Schema}."""
    base, donor, units = ABLATION_VARIANTS[variant]
    return swap_units(schemas[base], schemas[donor], units)


def _triples(lines):
    for line in lines:
        yield line[:-2].split(" ", 2)  # "<s> <p> o ." ; only the object can contain spaces


def build_graph(lines: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """
    (edge_index, edge_type, node_ids, relation_ids). Every triple becomes a forward edge
    (type rel*2) and an inverse edge (type rel*2+1), the convention of training_data.Ontology.
    rdf:type triples stay in: typing is part of how each KG is constructed.
    """
    node_ids, relation_ids, src, dst, rel = {}, {}, [], [], []
    for s, p, o in _triples(lines):
        s_id = node_ids.setdefault(s, len(node_ids))
        o_id = node_ids.setdefault(o, len(node_ids))
        r_id = relation_ids.setdefault(p, len(relation_ids))
        src += [s_id, o_id]
        dst += [o_id, s_id]
        rel += [r_id * 2, r_id * 2 + 1]
    return torch.tensor([src, dst]), torch.tensor(rel), node_ids, relation_ids


def literal_number(lexical: str, datatype: str) -> float | None:
    """Numeric reading of a literal: the number itself, booleans as 0/1, dates as a
    fractional year. None for strings."""
    if datatype in NUMERIC_DATATYPES:
        return float(lexical)
    if datatype == XSD_NS + "boolean":
        return {"true": 1.0, "false": 0.0}[lexical]
    if datatype == XSD_NS + "date":
        d = date.fromisoformat(lexical)
        days_in_year = (date(d.year + 1, 1, 1) - date(d.year, 1, 1)).days
        return d.year + (d.timetuple().tm_yday - 1) / days_in_year
    return None


def node_features(lines: list[str], node_ids: dict) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    (key_id, value, vocabulary) per node; see models.NodeEncoder.
      instance IRI   key "type:<sorted rdf:type IRIs>"      value 0
      class IRI      key "class:<IRI>" (an rdf:type object) value 0
      number / date  key "literal:<datatype>"               value z-scored over that datatype's literal nodes
      boolean        key "literal:<datatype>"               value 0 / 1
      string         key "string:<lexical form>"            value 0
    """
    types = {}
    for s, p, o in _triples(lines):
        if p == f"<{RDF_TYPE}>":
            types.setdefault(s, set()).add(o)
    classes = {t for ts in types.values() for t in ts}

    keys, values = [], []
    for node in node_ids:  # insertion order == node id
        literal = _LITERAL.match(node)
        if literal:
            lexical, datatype = literal.groups()
            number = literal_number(lexical, datatype)
            keys.append(f"literal:{datatype}" if number is not None else f"string:{lexical}")
            values.append(number if number is not None else 0.0)
        elif node in types:
            keys.append("type:" + "|".join(sorted(types[node])))
            values.append(0.0)
        elif node in classes:
            keys.append(f"class:{node}")
            values.append(0.0)
        else:
            raise ValueError(f"{node} is neither a literal, a typed instance nor a class")

    values = np.array(values)
    keys_arr = np.array(keys)
    for datatype in NUMERIC_DATATYPES | {XSD_NS + "date"}:
        on = keys_arr == f"literal:{datatype}"
        if on.any():
            std = values[on].std()
            values[on] = (values[on] - values[on].mean()) / std if std > 0 else 0.0

    vocabulary = sorted(set(keys))
    index = {k: i for i, k in enumerate(vocabulary)}
    return torch.tensor([index[k] for k in keys]), torch.tensor(values, dtype=torch.float), vocabulary


def degree_feature(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """log1p of the node degree (edges are doubled, so in-degree = degree), z-scored."""
    deg = torch.bincount(edge_index[1], minlength=num_nodes).float()
    log_deg = torch.log1p(deg)
    return (log_deg - log_deg.mean()) / (log_deg.std() + 1e-8)


@dataclass
class KGData:
    data: Data                 # edge_index, edge_type, y, key_id, value, degree_feat
    train: torch.Tensor        # node ids of the split's RaceEntries
    val: torch.Tensor
    test: torch.Tensor
    num_relations: int         # edge types, forward + inverse
    vocabulary: list[str]      # feature keys; models need len(vocabulary)
    node_ids: dict[str, int]   # N-Triples term -> node id


def build_kg_data(kg: str, entities: pd.DataFrame, assertions: pd.DataFrame,
                  split: tuple[list[str], list[str], list[str]], schema=None) -> KGData:
    """Masked graph, features, labels and split node ids for one KG."""
    schema = schema or read_schema(SCHEMA_FILES[kg])
    lines = masked_lines(entities, assertions, schema)
    edge_index, edge_type, node_ids, relation_ids = build_graph(lines)
    key_id, value, vocabulary = node_features(lines, node_ids)

    labels = race_entry_labels(assertions)
    y = torch.zeros(len(node_ids))
    y[[node_ids[f"<{iri}>"] for iri in labels.index]] = torch.tensor(labels.values, dtype=torch.float)
    ids = lambda iris: torch.tensor([node_ids[f"<{iri}>"] for iri in iris])

    data = Data(edge_index=edge_index, edge_type=edge_type, y=y, num_nodes=len(node_ids),
                key_id=key_id, value=value, degree_feat=degree_feature(edge_index, len(node_ids)))
    return KGData(data, *map(ids, split), 2 * len(relation_ids), vocabulary, node_ids)
