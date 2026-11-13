import random
import re

import pandas as pd
import pytest
import torch

from f1_node_data import (
    ABLATION_UNITS, ABLATION_VARIANTS, LABEL_PROPERTY, OUTCOME_PROPERTIES, _triples, ablation_schema,
    build_graph, build_kg_data, degree_feature, literal_number, masked_lines, node_features,
    race_entry_labels, stratified_split, swap_units,
)
from f1_population import NS, RDF_TYPE, SULO_NS, XSD_NS, render_ntriples

KGS = ("adhoc", "pure_sulo", "hybrid_sulo")
MASKED = OUTCOME_PROPERTIES | {LABEL_PROPERTY}
SPLIT = ([NS + "RaceEntry/100", NS + "RaceEntry/101"], [NS + "RaceEntry/102"], [NS + "RaceEntry/103"])


@pytest.fixture(scope="module")
def masked(built, schemas):
    entities, assertions = built
    return {kg: masked_lines(entities, assertions, schemas[kg]) for kg in KGS}


@pytest.fixture(scope="module")
def kg_data(built, schemas):
    entities, assertions = built
    return {kg: build_kg_data(kg, entities, assertions, SPLIT, schemas[kg]) for kg in KGS}


# ── Masking ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kg", KGS)
def test_no_label_or_outcome_survives_in_any_shape(masked, kg):
    # Covers direct predicates and Pure SULO's <subject>/<property> intermediate nodes.
    for line in masked[kg]:
        assert not any(prop in line for prop in MASKED), line
    assert not any('"Finished"' in line or '"Engine"' in line for line in masked[kg])   # status texts


def test_pure_keeps_a_start_position_that_equalled_the_finish_position(masked):
    # RaceEntry/101 started and finished P1: in Pure both are one hasFeature triple.
    assert f"<{NS}RaceEntry/101> <{SULO_NS}hasFeature> <{NS}Position/1> ." in masked["pure_sulo"]


@pytest.mark.parametrize("kg", KGS)
def test_masking_removes_only_the_masked_properties(built, schemas, masked, kg):
    entities, assertions = built
    kept = assertions[~assertions["property"].isin(MASKED)]
    assert masked[kg] == render_ntriples(entities, kept, schemas[kg])[0]
    assert len(masked[kg]) < len(render_ntriples(entities, assertions, schemas[kg])[0])


# ── Labels and split ──────────────────────────────────────────────────────

def test_labels_come_from_the_assertions(built):
    labels = race_entry_labels(built[1])
    assert labels.to_dict() == {NS + "RaceEntry/100": 1, NS + "RaceEntry/101": 1,
                                NS + "RaceEntry/102": 0, NS + "RaceEntry/103": 1}   # 104: points missing


@pytest.fixture
def labels():
    rng = random.Random(0)
    return pd.Series([int(rng.random() < 0.313) for _ in range(1000)], index=[f"e{i}" for i in range(1000)])


def test_split_is_disjoint_complete_and_80_10_10(labels):
    train, val, test = stratified_split(labels)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    assert set(train) | set(val) | set(test) == set(labels.index)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


def test_split_is_stratified(labels):
    rate = labels.mean()
    for part in stratified_split(labels):
        assert abs(labels[part].sum() - rate * len(part)) <= 1


def test_split_is_reproducible(labels):
    assert stratified_split(labels) == stratified_split(labels)
    assert stratified_split(labels) != stratified_split(labels, seed=7)


@pytest.mark.parametrize("kg", KGS)
def test_every_kg_gets_the_same_labelled_split(kg_data, kg):
    d = kg_data[kg]
    iri_of = {v: k for k, v in d.node_ids.items()}
    names = lambda ids: [iri_of[i] for i in ids.tolist()]
    assert names(d.train) == [f"<{NS}RaceEntry/100>", f"<{NS}RaceEntry/101>"]
    assert names(d.val) == [f"<{NS}RaceEntry/102>"] and names(d.test) == [f"<{NS}RaceEntry/103>"]
    assert d.data.y[torch.cat([d.train, d.val, d.test])].tolist() == [1, 1, 0, 1]
    assert d.data.y.sum() == 3                                  # no label on any other node


# ── Features ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("lexical, datatype, number", [
    ("78.050", "decimal", 78.05),
    ("12", "integer", 12.0),
    ("0", "nonNegativeInteger", 0.0),
    ("true", "boolean", 1.0),
    ("false", "boolean", 0.0),
    ("2024-01-01", "date", 2024.0),
    ("2024-12-31", "date", 2024 + 365 / 366),                  # leap year
    ("2023-12-31", "date", 2023 + 364 / 365),
    ("Finished", "string", None),
])
def test_literal_number(lexical, datatype, number):
    assert literal_number(lexical, XSD_NS + datatype) == pytest.approx(number) if number is not None \
        else literal_number(lexical, XSD_NS + datatype) is None


def _keys(lines):
    _, _, node_ids, _ = build_graph(lines)
    key_id, value, vocabulary = node_features(lines, node_ids)
    return {node: (vocabulary[k], v) for node, k, v in zip(node_ids, key_id.tolist(), value.tolist())}


@pytest.mark.parametrize("kg", KGS)
def test_all_race_entries_share_one_key(masked, kg):
    keys = _keys(masked[kg])
    entry_keys = {k for node, (k, _) in keys.items() if re.fullmatch(rf"<{NS}RaceEntry/\d+>", node)}
    assert len(entry_keys) == 1
    expected = {"adhoc": f"type:<{NS}RaceEntry>",
                "pure_sulo": f"type:<{NS}RaceEntry>|<{SULO_NS}Process>",
                "hybrid_sulo": f"type:<{NS}RaceEntry>|<{SULO_NS}Process>"}[kg]
    assert entry_keys == {expected}


def test_key_kinds(masked):
    keys = _keys(masked["pure_sulo"])
    assert keys[f"<{NS}RaceEntry>"][0] == f"class:<{NS}RaceEntry>"
    assert keys[f"<{NS}Race/10/hasRound>"][0] == f"type:<{SULO_NS}Quantity>"         # intermediate node
    assert keys[f"<{NS}Circuit/1/hasCountry>"][0] == keys[f"<{NS}Circuit/2/hasCountry>"][0]
    assert keys[f'"Bahrain"^^<{XSD_NS}string>'][0] == 'string:Bahrain'
    assert keys[f'"25"^^<{XSD_NS}integer>'][0] == f"literal:{XSD_NS}integer"


@pytest.mark.parametrize("kg", KGS)
def test_numeric_values_are_standardised_per_datatype(masked, kg):
    keys = _keys(masked[kg])
    for datatype in ("integer", "decimal", "date"):
        values = [v for k, v in keys.values() if k == f"literal:{XSD_NS}{datatype}"]
        if len(values) > 1:
            assert abs(sum(values) / len(values)) < 1e-6
    integers = {node: v for node, (k, v) in keys.items() if k == f"literal:{XSD_NS}integer"}
    assert integers[f'"2013"^^<{XSD_NS}integer>'] > integers[f'"2009"^^<{XSD_NS}integer>']   # order kept


@pytest.mark.parametrize("kg", KGS)
def test_features_never_read_iri_text(masked, kg):
    # Rename every instance IRI (incl. Pure's <subject>/<property> nodes) to an opaque one:
    # the multiset of node features must not change.
    lines = masked[kg]
    instances = sorted({m for line in lines for m in re.findall(rf"<{re.escape(NS)}[^>]*/[^>]*>", line)})
    opaque = dict(zip(instances, random.Random(0).sample([f"<urn:node:{i}>" for i in range(len(instances))],
                                                         len(instances))))
    renamed = [re.sub(rf"<{re.escape(NS)}[^>]*/[^>]*>", lambda m: opaque[m.group(0)], line) for line in lines]
    assert sorted(_keys(lines).values()) == sorted(_keys(renamed).values())


def test_untyped_instance_is_rejected():
    lines = [f"<{NS}RaceEntry/1> <{NS}hasDriver> <{NS}Driver/1> ."]
    _, _, node_ids, _ = build_graph(lines)
    with pytest.raises(ValueError):
        node_features(lines, node_ids)


# ── Graph tensors ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("kg", KGS)
def test_every_triple_is_a_forward_and_an_inverse_edge(masked, kg_data, kg):
    d = kg_data[kg]
    predicates = {line.split(" ", 2)[1] for line in masked[kg]}
    assert d.num_relations == 2 * len(predicates)
    assert d.data.edge_index.size(1) == 2 * len(masked[kg])
    forward, inverse = d.data.edge_index[:, ::2], d.data.edge_index[:, 1::2]
    assert torch.equal(forward, inverse.flip(0))
    assert torch.equal(d.data.edge_type[1::2], d.data.edge_type[::2] + 1)


@pytest.mark.parametrize("kg", KGS)
def test_degree_is_computed_on_the_masked_graph(masked, kg_data, kg):
    edge_index, *_ = build_graph(masked[kg])
    assert torch.equal(kg_data[kg].data.degree_feat, degree_feature(edge_index, kg_data[kg].data.num_nodes))


def test_degree_feature_is_standardised():
    edge_index = torch.tensor([[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]])
    deg = degree_feature(edge_index, 4)
    assert deg.mean().abs() < 1e-6 and (deg.std() - 1).abs() < 1e-4
    assert deg[0] > deg[1] == deg[2] == deg[3]


# ── Enrichment ablation variants ──────────────────────────────────────────


def _up_to_predicate_names(lines):
    """A graph with its predicate IRIs forgotten: each node's feature key and value, and the
    multiset of per-predicate (subject, object) edge sets."""
    edges = {}
    for s, p, o in _triples(lines):
        edges.setdefault(p, set()).add((s, o))
    _, _, node_ids, _ = build_graph(lines)
    key_id, value, vocabulary = node_features(lines, node_ids)
    nodes = {n: (vocabulary[k], v) for n, k, v in zip(node_ids, key_id.tolist(), value.tolist())}
    return nodes, sorted(sorted(e) for e in edges.values())


def _predicates(lines):
    return {p for _, p, _ in _triples(lines)}


@pytest.mark.parametrize("base, other", [("hybrid_sulo", "adhoc"), ("adhoc", "hybrid_sulo")])
def test_swapping_every_unit_turns_one_kg_into_the_other(built, schemas, base, other):
    # Hybrid minus all units is Ad-hoc and Ad-hoc plus all units is Hybrid, up to the
    # heldAtCircuit/isInCircuit predicate rename, which is therefore not a unit.
    entities, assertions = built
    swapped = masked_lines(entities, assertions, swap_units(schemas[base], schemas[other], ABLATION_UNITS))
    target = masked_lines(entities, assertions, schemas[other])
    assert _up_to_predicate_names(swapped) == _up_to_predicate_names(target)
    assert _predicates(swapped) ^ _predicates(target) == {f"<{NS}heldAtCircuit>", f"<{SULO_NS}isIn>"}


PROPERTY_UNITS = [u for u in ABLATION_UNITS if u != "sulo_typing"]


@pytest.mark.parametrize("unit", PROPERTY_UNITS)
@pytest.mark.parametrize("direction", ["hybrid_minus", "adhoc_plus"])
def test_a_property_unit_changes_only_its_own_triples(built, schemas, unit, direction):
    entities, assertions = built
    variant = f"{direction}_{unit}"
    base, donor, _ = ABLATION_VARIANTS[variant]
    own = assertions[assertions["property"].isin(ABLATION_UNITS[unit])]
    assert not own.empty
    shape = lambda kg: set(render_ntriples(entities.iloc[:0], own, schemas[kg])[0])   # the unit's triples in kg
    base_lines = set(masked_lines(entities, assertions, schemas[base]))
    variant_lines = set(masked_lines(entities, assertions, ablation_schema(variant, schemas)))
    assert base_lines - variant_lines == shape(base)
    assert variant_lines - base_lines == shape(donor)
    assert shape("hybrid_sulo") and shape("hybrid_sulo") != shape("adhoc")


def test_rank_reification_unit_moves_the_rank_behind_an_intermediate_node(built, schemas):
    entities, assertions = built
    literal = f'"1"^^<{XSD_NS}nonNegativeInteger> .'
    direct = f"<{NS}Position/1> <{NS}hasRank> {literal}"
    reified = f"<{NS}Position/1/hasRank> <{SULO_NS}hasValue> {literal}"
    minus = masked_lines(entities, assertions, ablation_schema("hybrid_minus_rank_reification", schemas))
    plus = masked_lines(entities, assertions, ablation_schema("adhoc_plus_rank_reification", schemas))
    assert direct in minus and reified not in minus
    assert reified in plus and direct not in plus


@pytest.mark.parametrize("variant", ["hybrid_minus_sulo_typing", "adhoc_plus_sulo_typing"])
def test_typing_unit_changes_only_the_sulo_superclass_types(built, schemas, variant):
    entities, assertions = built
    base, _, _ = ABLATION_VARIANTS[variant]
    no_facts = assertions.iloc[:0]
    typing = (set(render_ntriples(entities, no_facts, schemas["hybrid_sulo"])[0])
              - set(render_ntriples(entities, no_facts, schemas["adhoc"])[0]))
    assert typing and all(f"<{RDF_TYPE}> <{SULO_NS}" in line for line in typing)
    base_lines = set(masked_lines(entities, assertions, schemas[base]))
    variant_lines = set(masked_lines(entities, assertions, ablation_schema(variant, schemas)))
    removed, added = (typing, set()) if base == "hybrid_sulo" else (set(), typing)
    assert base_lines - variant_lines == removed
    assert variant_lines - base_lines == added


def test_every_control_is_its_own_kg(schemas):
    for kg in ("adhoc", "hybrid_sulo"):
        assert ablation_schema(kg, schemas) == schemas[kg]


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_every_variant_keeps_the_mask_and_the_labelled_split(built, schemas, kg_data, variant):
    entities, assertions = built
    schema = ablation_schema(variant, schemas)
    for line in masked_lines(entities, assertions, schema):
        assert not any(prop in line for prop in MASKED), line
    d = build_kg_data(ABLATION_VARIANTS[variant][0], entities, assertions, SPLIT, schema)
    ref = kg_data["adhoc"]
    for part in ("train", "val", "test"):
        assert torch.equal(d.data.y[getattr(d, part)], ref.data.y[getattr(ref, part)])
