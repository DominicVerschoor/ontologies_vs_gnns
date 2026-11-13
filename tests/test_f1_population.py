from decimal import Decimal

import numpy as np
import pytest
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from f1_population import (
    EXPECTED_SET_ASIDE, NS, SULO_NS, best_lap, lap_time_to_seconds,
    parse_rank, points_scored, practice_lap_time_to_seconds, read_schema, render_ntriples, subset,
)

NAN = np.nan
F1 = Namespace(NS)
SULO = Namespace(SULO_NS)


# ── Value conversions ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text, seconds", [
    ("1:27.452", Decimal("87.452")),
    ("0:59.999", Decimal("59.999")),
    ("1:59.999", Decimal("119.999")),
    ("2:00.000", Decimal("120.000")),
])
def test_lap_time_to_seconds(text, seconds):
    assert lap_time_to_seconds(text) == seconds


@pytest.mark.parametrize("missing", [None, NAN])
def test_lap_time_missing_is_none(missing):
    assert lap_time_to_seconds(missing) is None


@pytest.mark.parametrize("bad", ["87.452", "1:7.452", "1:60.000", "1:75.000", "DNF"])
def test_lap_time_rejects_unknown_format(bad):
    with pytest.raises(ValueError):
        lap_time_to_seconds(bad)


@pytest.mark.parametrize("text, seconds", [
    ("0 days 00:01:17.252000", Decimal("77.252")),
    ("0 days 00:01:31", Decimal("91.000")),             # whole second, no fraction in the source
    ("0 days 00:00:59.999000", Decimal("59.999")),
    ("0 days 00:02:00.000000", Decimal("120.000")),
])
def test_practice_lap_time_to_seconds(text, seconds):
    # Equal lexical form, not just equal value: ms precision like qualifying times ("91.000", not "91")
    assert format(practice_lap_time_to_seconds(text), "f") == format(seconds, "f")


@pytest.mark.parametrize("missing", [None, NAN])
def test_practice_lap_time_missing_is_none(missing):
    assert practice_lap_time_to_seconds(missing) is None


@pytest.mark.parametrize("bad", ["1:17.252", "0 days 00:01:17.252500", "0 days 01:00:00.000000",
                                 "0 days 00:01:60.000000", "1 days 00:01:17.252000", "0 days 00:01:17.25"])
def test_practice_lap_time_rejects_unknown_format(bad):
    with pytest.raises(ValueError):
        practice_lap_time_to_seconds(bad)


def test_best_lap():
    assert best_lap("1:30.000", "1:29.500", NAN) == Decimal("89.500")
    assert best_lap(NAN, "1:31.000", NAN) == Decimal("91.000")
    assert best_lap(NAN, NAN, NAN) is None


@pytest.mark.parametrize("points, scored", [("0", False), ("0.0", False), ("0.5", True), ("25", True),
                                            (NAN, None), (None, None)])
def test_points_scored(points, scored):
    assert points_scored(points) is scored


@pytest.mark.parametrize("value, rank", [("1", 1), ("20", 20), ("0", None), (NAN, None), (None, None)])
def test_parse_rank(value, rank):
    assert parse_rank(value) == rank


# ── Schema reading ────────────────────────────────────────────────────────

def test_race_points_shape(schemas):
    for kg in ("adhoc", "hybrid_sulo"):
        shape = schemas[kg].properties["hasRacePoints"]
        assert (shape.predicate, shape.hop) == (NS + "hasRacePoints", None)
        assert shape.datatype.endswith("#boolean")
    shape = schemas["pure_sulo"].properties["hasRacePoints"]
    assert shape.predicate == SULO_NS + "hasFeature"
    assert shape.hop == (SULO_NS + "Quantity", SULO_NS + "hasValue")


def test_rank_is_sulo_grounded_in_pure_and_hybrid(schemas):
    for kg in ("pure_sulo", "hybrid_sulo"):
        shape = schemas[kg].properties["hasRank"]
        assert shape.predicate == SULO_NS + "isReferredToIn"
        assert shape.hop == (SULO_NS + "InformationObject", SULO_NS + "hasValue")


def test_driver_role_chain_in_pure(schemas):
    shape = schemas["pure_sulo"].properties["hasDriver"]
    assert shape.predicate == SULO_NS + "hasParticipant"
    assert shape.hop == (SULO_NS + "Role", SULO_NS + "isFeatureOf")
    assert shape.datatype is None


def test_hybrid_sulo_predicates(schemas):
    props = schemas["hybrid_sulo"].properties
    assert props["hasTeamMember"].predicate == SULO_NS + "hasItem"
    assert props["precedesRace"].predicate == SULO_NS + "precedes"
    assert props["isInCircuit"].predicate == SULO_NS + "isIn"


def test_sulo_superclasses(schemas):
    assert schemas["adhoc"].sulo_superclass == {}
    assert schemas["pure_sulo"].sulo_superclass["Team"] == SULO_NS + "Collection"   # sulo:hasItem's domain
    assert schemas["hybrid_sulo"].sulo_superclass["Team"] == SULO_NS + "Collection"
    assert schemas["pure_sulo"].sulo_superclass["Race"] == SULO_NS + "Process"


def test_schemas_agree_on_literal_vs_object(schemas):
    shared = set.intersection(*(set(s.properties) for s in schemas.values()))
    for prop in shared:
        datatypes = {s.properties[prop].datatype for s in schemas.values()}
        assert len(datatypes) == 1, f"{prop} has datatypes {datatypes}"


_TTL_HEADER = f"""@prefix : <{NS}> . @prefix sulo: <{SULO_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


@pytest.mark.parametrize("body", [
    ":Team rdfs:subClassOf sulo:Collection, sulo:InformationObject .",               # two SULO superclasses
    ":hasYear rdfs:domain :Season, :Race ; rdfs:range xsd:integer .",               # two domains
    ":hasYear rdfs:domain :Season .",                                               # no range
])
def test_read_schema_rejects_ambiguous_schemas(tmp_path, body):
    path = tmp_path / "schema.ttl"
    path.write_text(_TTL_HEADER + body)
    with pytest.raises(AssertionError):
        read_schema(str(path))


# ── Rendering a hand-built fixture ────────────────────────────────────────

def _parse(lines):
    return Graph().parse(data="\n".join(lines), format="nt")


@pytest.fixture(scope="module")
def rendered(built, schemas):
    """kg -> (N-Triples lines, set-aside properties)"""
    return {kg: render_ntriples(*built, schema) for kg, schema in schemas.items()}


@pytest.fixture(scope="module")
def graphs(rendered):
    return {kg: _parse(lines) for kg, (lines, _) in rendered.items()}


def test_scope_keeps_every_season(built):
    nodes = set(built[0]["iri"])
    assert {NS + "RaceEntry/100", NS + "Race/1", NS + "Race/3", NS + "Season/2009"} <= nodes


def test_scope_excludes_indy500_shared_drives_and_orphan_sessions(built):
    nodes = {iri.removeprefix(NS) for iri in built[0]["iri"]}
    assert {"Race/2", "RaceEntry/105", "Circuit/3", "Driver/4",                        # Indianapolis 500
            "RaceEntry/106", "RaceEntry/107", "Driver/5", "Team/2",                    # shared drive
            "SessionResult/Qualifying_502", "SessionResult/FP2_10_3", "Driver/3",      # no race entry
            }.isdisjoint(nodes)


def test_every_object_iri_is_a_typed_entity(built, schemas):
    entities, assertions = built
    object_props = {p for p, s in schemas["hybrid_sulo"].properties.items() if s.datatype is None}
    object_props |= {"heldAtCircuit"}
    objects = set(assertions.loc[assertions["property"].isin(object_props), "value"])
    assert objects <= set(entities["iri"])


def test_set_aside_matches_expected(rendered):
    for kg, (_, set_aside) in rendered.items():
        assert set_aside == EXPECTED_SET_ASIDE[kg]


def test_domain_mismatch_raises(built, schemas):
    entities, assertions = built
    wrong = assertions.head(1).assign(property="hasRacePoints", subject_class="Driver")
    with pytest.raises(ValueError):
        render_ntriples(entities, wrong, schemas["adhoc"])


def test_position_nodes_are_shared(graphs):
    g = graphs["adhoc"]
    assert (F1["RaceEntry/101"], F1.hasFinishPosition, F1["Position/1"]) in g
    assert (F1["RaceEntry/103"], F1.hasFinishPosition, F1["Position/1"]) in g
    assert len(list(g.triples((F1["Position/1"], F1.hasRank, None)))) == 1
    assert len(list(graphs["pure_sulo"].triples((F1["Position/1"], None, F1["Position/1/hasRank"])))) == 1


def test_missing_values_are_omitted(graphs):
    g = graphs["adhoc"]
    assert (F1["RaceEntry/102"], F1.hasStartPosition, None) not in g      # grid 0
    assert (F1["RaceEntry/102"], F1.hasFinishPosition, None) not in g     # unclassified
    assert (F1["RaceEntry/104"], F1.hasRacePoints, None) not in g         # points missing
    assert (F1["Driver/2"], F1.hasDateOfBirth, None) not in g
    assert (F1["SessionResult/Qualifying_501"], F1.hasBestLapTime, None) not in g


def test_session_results(graphs):
    g = graphs["adhoc"]
    quali, sprint = F1["SessionResult/Qualifying_500"], F1["SessionResult/Sprint_600"]
    assert (F1["RaceEntry/101"], F1.hasSessionResult, quali) in g
    assert (F1["RaceEntry/103"], F1.hasSessionResult, sprint) in g
    assert g.value(quali, F1.hasBestLapTime) == Literal(Decimal("89.500"))
    assert (quali, F1.hasSessionPoints, None) not in g                    # qualifying awards none
    assert g.value(sprint, F1.hasSessionPoints) == Literal(True)
    assert g.value(sprint, F1.hasSessionLapsCompleted) == Literal(20)


def test_practice_session_results(graphs):
    g = graphs["adhoc"]
    fp1, fp1_whole_second = F1["SessionResult/FP1_10_1"], F1["SessionResult/FP1_10_2"]
    assert (F1["RaceEntry/101"], F1.hasSessionResult, fp1) in g
    assert (F1["RaceEntry/102"], F1.hasSessionResult, fp1_whole_second) in g
    assert str(g.value(fp1, F1.hasSessionType)) == "FP1"
    assert (fp1, F1.hasPosition, F1["Position/2"]) in g                   # shared rank node
    assert str(g.value(fp1, F1.hasBestLapTime)) == "90.500"
    assert str(g.value(fp1_whole_second, F1.hasBestLapTime)) == "91.000"
    assert g.value(fp1, F1.hasSessionLapsCompleted) == Literal(25)
    assert (fp1, F1.hasSessionPoints, None) not in g                      # practice awards none


def test_literal_escaping_round_trips(graphs):
    g = graphs["adhoc"]
    assert g.value(F1["Driver/2"], F1.hasDriverName).toPython() == 'Jean "JJ" Back\\slash'
    assert g.value(F1["Circuit/2"], F1.hasLocation).toPython() == "Mel\nbourne\r"


def test_pure_uses_only_sulo_and_type_predicates(graphs):
    g = graphs["pure_sulo"]
    assert all(str(p).startswith(SULO_NS) or p == RDF.type for p in g.predicates())
    label_node = F1["RaceEntry/101/hasRacePoints"]
    assert (F1["RaceEntry/101"], SULO.hasFeature, label_node) in g
    assert (label_node, SULO.hasValue, Literal(True)) in g
    assert (F1["RaceEntry/101/hasDriver"], SULO.isFeatureOf, F1["Driver/1"]) in g


def test_sulo_superclass_typing(graphs):
    for kg in ("pure_sulo", "hybrid_sulo"):
        assert (F1["Race/10"], RDF.type, SULO.Process) in graphs[kg]
    assert not any(str(o).startswith(SULO_NS) for o in graphs["adhoc"].objects(None, RDF.type))


def test_hybrid_derived_relations(graphs):
    g = graphs["hybrid_sulo"]
    assert (F1["Race/10"], SULO.precedes, F1["Race/11"]) in g
    assert (F1["Race/11"], SULO.precedes, None) not in g          # last race of its season
    assert (F1["Race/1"], SULO.precedes, F1["Race/3"]) in g       # chains past the excluded Indy 500
    assert (F1["Season/2009"], SULO.precedes, F1["Season/2010"]) in g
    assert (F1["Season/2010"], SULO.precedes, F1["Season/2011"]) in g
    assert (F1["Season/2011"], SULO.precedes, None) not in g      # 2012 is missing
    assert (F1["Team/1"], SULO.hasItem, F1["Driver/2"]) in g
    assert (F1["Team/1"], SULO.hasItem, F1["Driver/5"]) not in g  # shared-drive rows are dropped
    assert (F1["Race/10"], SULO.isIn, F1["Circuit/1"]) in g


def test_sulo_enrichments_in_pure_but_not_adhoc(graphs):
    g = graphs["pure_sulo"]
    assert (F1["Race/10"], SULO.precedes, F1["Race/11"]) in g
    assert (F1["Season/2010"], SULO.precedes, F1["Season/2011"]) in g
    assert (F1["Team/1"], SULO.hasItem, F1["Driver/2"]) in g
    enrichments = {F1.precedesRace, F1.precedesSeason, F1.hasTeamMember, F1.isInCircuit, SULO.precedes, SULO.hasItem}
    assert enrichments.isdisjoint(graphs["adhoc"].predicates())


def _labels(graph):
    """RaceEntry IRI -> hasRacePoints value, for both the direct and the Pure SULO path."""
    labels = {}
    for s, p, o in graph:
        if p == F1.hasRacePoints:
            labels[str(s)] = o.toPython()
        elif p == SULO.hasValue and str(s).endswith("/hasRacePoints"):
            labels[str(s)[: -len("/hasRacePoints")]] = o.toPython()
    return labels


def test_all_kgs_have_the_same_entries_and_labels(graphs):
    labels = [_labels(g) for g in graphs.values()]
    assert labels[0] == {NS + "RaceEntry/100": True, NS + "RaceEntry/101": True, NS + "RaceEntry/102": False,
                         NS + "RaceEntry/103": True}
    assert labels[0] == labels[1] == labels[2]


def test_pure_merges_equal_start_and_finish_but_masking_keeps_start(built, schemas):
    entities, assertions = built
    full = _parse(render_ntriples(entities, assertions, schemas["pure_sulo"])[0])
    assert list(full.objects(F1["RaceEntry/101"], SULO.hasFeature)).count(F1["Position/1"]) == 1

    masked_assertions = assertions[assertions["property"] != "hasFinishPosition"]
    masked = _parse(render_ntriples(entities, masked_assertions, schemas["pure_sulo"])[0])
    assert (F1["RaceEntry/101"], SULO.hasFeature, F1["Position/1"]) in masked   # start position survives
    assert (F1["RaceEntry/103"], SULO.hasFeature, F1["Position/2"]) in masked
    assert (F1["RaceEntry/103"], SULO.hasFeature, F1["Position/1"]) not in masked


def test_subset_follows_object_links_but_not_stop_properties(built):
    entities, assertions = built
    sub_entities, sub_assertions = subset(entities, assertions, [NS + "Race/10"])
    nodes = {iri.removeprefix(NS) for iri in sub_entities["iri"]}
    assert nodes == {"Race/10", "Season/2010", "Circuit/1", "RaceEntry/101", "RaceEntry/102",
                     "Driver/1", "Driver/2", "Team/1", "Position/1", "Position/2",
                     "SessionResult/Qualifying_500", "SessionResult/Qualifying_501",
                     "SessionResult/FP1_10_1", "SessionResult/FP1_10_2"}
    assert set(sub_assertions["subject"]) <= set(sub_entities["iri"])
    # precedesRace is kept as a reference, but Race/11 itself is not described
    assert (sub_assertions["property"] == "precedesRace").sum() == 1
    assert NS + "Race/11" not in set(sub_assertions["subject"])
