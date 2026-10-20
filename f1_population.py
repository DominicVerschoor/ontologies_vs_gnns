"""
Populates the three F1 knowledge graphs (adhoc / pure_sulo / hybrid_sulo) from the Kaggle
Formula 1 World Championship tables.

The data is turned once into KG-independent assertions (subject, subject class, schema
property, value). Each schema .ttl then decides how an assertion is written as triples:
a direct edge, or a SULO path through an intermediate node. All three KGs therefore
contain the same base facts by construction, and no triple shape is written by hand.
"""
import re
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
from rdflib import OWL, RDFS, Graph, URIRef
from rdflib.collection import Collection

NS = "https://example.org/f1-ontology/"
SULO_NS = "https://w3id.org/sulo/"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

SCHEMA_FILES = {
    "adhoc": "turtles/f1/base_kg.ttl",
    "pure_sulo": "turtles/f1/pure_sulo_kg.ttl",
    "hybrid_sulo": "turtles/f1/hybrid_sulo_kg.ttl",
}

# Assertions are built for the union of all three schemas' properties; each KG declares
# only some of them. Anything else a schema lacks is a mapping or schema error.
EXPECTED_SET_ASIDE = {
    "adhoc": {"isInCircuit", "precedesRace", "precedesSeason", "hasTeamMember"},
    "pure_sulo": {"isInCircuit"},
    "hybrid_sulo": {"heldAtCircuit"},
}


# ── Value conversions ─────────────────────────────────────────────────────

_LAP_TIME = re.compile(r"^(\d+):([0-5]\d\.\d+)$")
# FastF1 timedelta strings in practice_results.csv: millisecond precision, fraction omitted on whole seconds.
_PRACTICE_LAP_TIME = re.compile(r"^0 days 00:(\d\d):([0-5]\d)(?:\.(\d{3})000)?$")


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def lap_time_to_seconds(value) -> Decimal | None:
    """'1:27.452' -> Decimal('87.452'); missing -> None. Any other format raises."""
    if _missing(value):
        return None
    match = _LAP_TIME.match(value)
    if match is None:
        raise ValueError(f"Unrecognised lap time: {value!r}")
    return int(match.group(1)) * 60 + Decimal(match.group(2))


def practice_lap_time_to_seconds(value) -> Decimal | None:
    """
    '0 days 00:01:17.252000' -> Decimal('77.252'); '0 days 00:01:31' -> Decimal('91.000'); missing -> None.
    Always three decimals, so equal times get the same lexical form as qualifying times. Any other format raises.
    """
    if _missing(value):
        return None
    match = _PRACTICE_LAP_TIME.match(value)
    if match is None:
        raise ValueError(f"Unrecognised practice lap time: {value!r}")
    minutes, seconds, millis = match.groups()
    return Decimal(f"{int(minutes) * 60 + int(seconds)}.{millis or '000'}")


def best_lap(*lap_times) -> Decimal | None:
    """Fastest of the given lap times in seconds; None if all are missing."""
    seconds = [s for s in map(lap_time_to_seconds, lap_times) if s is not None]
    return min(seconds) if seconds else None


def points_scored(points) -> bool | None:
    """True if any championship points were awarded (the node-prediction label rule); missing -> None."""
    return None if _missing(points) else float(points) > 0


def parse_rank(value) -> int | None:
    """Classified rank; missing (unclassified) and 0 (pit-lane start) -> None."""
    if _missing(value):
        return None
    rank = int(value)
    return rank if rank > 0 else None


# ── Schema reading ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PropertyShape:
    domain: str                        # domain class IRI
    predicate: str                     # predicate leaving the subject
    hop: tuple[str, str] | None        # (intermediate node class, predicate to the value) for SULO paths
    datatype: str | None               # literal datatype IRI; None for object properties


@dataclass(frozen=True)
class Schema:
    properties: dict[str, PropertyShape]  # property local name -> shape
    sulo_superclass: dict[str, str]       # class local name -> its direct SULO superclass IRI


def _local(term) -> str:
    return str(term)[len(NS):]


def read_schema(ttl_path: str) -> Schema:
    """
    Reads each property's triple shape from a schema file.

    In the OWL schemas a property's rdfs:domain is a class equivalent to
    (DomainClass and (p1 some X)), where X is a class, a datatype, or
    (IntermediateClass and (p2 some Y)) for SULO value/role patterns. p1 (and p2) are
    the predicates the instance data uses. base_kg.ttl has plain RDFS domains, so the
    property itself is the predicate.
    """
    g = Graph().parse(ttl_path, format="turtle")
    properties = {}
    for prop in set(g.subjects(RDFS.domain)):
        domains, ranges = list(g.objects(prop, RDFS.domain)), list(g.objects(prop, RDFS.range))
        assert len(domains) == 1 and len(ranges) == 1, f"{prop}: needs exactly one rdfs:domain and rdfs:range"
        domain_class, predicate, hop = domains[0], prop, None
        expression = g.value(domain_class, OWL.equivalentClass)
        if expression is not None:
            domain_class, restriction = Collection(g, g.value(expression, OWL.intersectionOf))
            predicate = g.value(restriction, OWL.onProperty)
            assert isinstance(predicate, URIRef), f"{prop}: unexpected restriction property {predicate}"
            inner = g.value(g.value(restriction, OWL.someValuesFrom), OWL.intersectionOf)
            if inner is not None:
                hop_class, hop_restriction = Collection(g, inner)
                hop = (str(hop_class), str(g.value(hop_restriction, OWL.onProperty)))
        datatype = str(ranges[0]) if str(ranges[0]).startswith(XSD_NS) else None
        properties[_local(prop)] = PropertyShape(str(domain_class), str(predicate), hop, datatype)

    sulo_superclass = {}
    for cls, sup in g.subject_objects(RDFS.subClassOf):
        if str(cls).startswith(NS) and str(sup).startswith(SULO_NS):
            assert _local(cls) not in sulo_superclass, f"{cls}: more than one SULO superclass"
            sulo_superclass[_local(cls)] = str(sup)
    return Schema(properties, sulo_superclass)


# ── Assertions ────────────────────────────────────────────────────────────

def iri(cls: str, key) -> str:
    return f"{NS}{cls}/{key}"


def position_iri(rank) -> str | None:
    return iri("Position", rank) if rank is not None else None


def _scope(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # The Indianapolis 500 (1950-60) counted for the championship but was run to separate rules
    # with a separate field: 103 of its drivers never started another race.
    races = tables["races"][tables["races"]["name"] != "Indianapolis 500"].copy()
    results = tables["results"][tables["results"]["raceId"].isin(races["raceId"])]
    # Shared drives (one driver with two results in a race, up to 1978) are dropped entirely:
    # a RaceEntry is identified by (race, driver), which is also the session-result join key.
    results = results[~results.duplicated(["raceId", "driverId"], keep=False)].copy()
    entry_key = results[["raceId", "driverId", "resultId"]].rename(columns={"resultId": "entryId"})
    # Inner join: session results without a race entry (e.g. did not start the race) are dropped.
    qualifying = tables["qualifying"].merge(entry_key, on=["raceId", "driverId"], how="inner")
    sprint = tables["sprint_results"].merge(entry_key, on=["raceId", "driverId"], how="inner")
    practice = tables["practice_results"].merge(entry_key, on=["raceId", "driverId"], how="inner")

    races["round_int"] = races["round"].astype(int)
    races = races.sort_values(["year", "round_int"])
    races["nextRaceId"] = races.groupby("year")["raceId"].shift(-1)

    years = set(races["year"])
    seasons = tables["seasons"][tables["seasons"]["year"].isin(years)].copy()
    seasons["nextYear"] = (seasons["year"].astype(int) + 1).astype(str)
    seasons.loc[~seasons["nextYear"].isin(years), "nextYear"] = None

    status = tables["status"].set_index("statusId")["status"]
    results["statusText"] = results["statusId"].map(status)

    return {
        "seasons": seasons,
        "races": races,
        "circuits": tables["circuits"][tables["circuits"]["circuitId"].isin(races["circuitId"])],
        "drivers": tables["drivers"][tables["drivers"]["driverId"].isin(results["driverId"])],
        "constructors": tables["constructors"][tables["constructors"]["constructorId"].isin(results["constructorId"])],
        "results": results,
        "qualifying": qualifying,
        "sprint": sprint,
        "practice": practice,
        "team_members": results[["constructorId", "driverId"]].drop_duplicates(),
    }


def _iris(cls: str, keys: pd.Series) -> pd.Series:
    return keys.map(lambda k: iri(cls, k), na_action="ignore")


def _positions(ranks: pd.Series) -> pd.Series:
    # Straight from source string to IRI: an int/None intermediate Series would be cast to float.
    return ranks.map(lambda v: position_iri(parse_rank(v)))


def _flags(points: pd.Series) -> pd.Series:
    """xsd:boolean lexical form of points_scored; missing stays missing."""
    return points.map(lambda p: {True: "true", False: "false", None: None}[points_scored(p)])


def _decimal_str(value) -> str | None:
    return None if value is None else format(value, "f")


def property_sources(f: dict[str, pd.DataFrame]) -> list[tuple[str, pd.Series, dict]]:
    """
    Every (subject class, subject IRIs, {schema property: values}) mapping, with subjects and
    values aligned by position. Values are literal lexical forms or object IRIs; None/NaN means
    the triple is omitted.
    """
    seasons, races, results = f["seasons"], f["races"], f["results"]
    circuits, drivers, constructors = f["circuits"], f["drivers"], f["constructors"]
    quali, sprint, practice, members = f["qualifying"], f["sprint"], f["practice"], f["team_members"]
    quali_iri = _iris("SessionResult", "Qualifying_" + quali["qualifyId"])
    sprint_iri = _iris("SessionResult", "Sprint_" + sprint["resultId"])
    # practice_results.csv has no id column; (session, race, driver) is its key.
    practice_iri = _iris("SessionResult", practice["session"] + "_" + practice["raceId"] + "_" + practice["driverId"])
    race_circuit = _iris("Circuit", races["circuitId"])
    rank_columns = [results["grid"], results["position"], quali["position"], sprint["position"], practice["position"]]
    ranks = pd.Series(sorted({r for col in rank_columns for r in map(parse_rank, col) if r is not None}), dtype=object)

    return [
        ("Season", _iris("Season", seasons["year"]), {
            "hasYear": seasons["year"],
            "precedesSeason": _iris("Season", seasons["nextYear"]),
        }),
        ("Race", _iris("Race", races["raceId"]), {
            "hasRaceName": races["name"],
            "hasRaceDate": races["date"],
            "hasRound": races["round"],
            "partOfSeason": _iris("Season", races["year"]),
            "heldAtCircuit": race_circuit,
            "isInCircuit": race_circuit,
            "precedesRace": _iris("Race", races["nextRaceId"]),
        }),
        ("Race", _iris("Race", results["raceId"]), {
            "hasRaceEntry": _iris("RaceEntry", results["resultId"]),
        }),
        ("Circuit", _iris("Circuit", circuits["circuitId"]), {
            "hasCircuitName": circuits["name"],
            "hasLocation": circuits["location"],
            "hasCountry": circuits["country"],
        }),
        ("Driver", _iris("Driver", drivers["driverId"]), {
            "hasDriverName": drivers["forename"] + " " + drivers["surname"],
            "hasDateOfBirth": drivers["dob"],
            "hasNationality": drivers["nationality"],
        }),
        ("Team", _iris("Team", constructors["constructorId"]), {
            "hasTeamName": constructors["name"],
            "hasTeamNationality": constructors["nationality"],
        }),
        ("Team", _iris("Team", members["constructorId"]), {
            "hasTeamMember": _iris("Driver", members["driverId"]),
        }),
        ("RaceEntry", _iris("RaceEntry", results["resultId"]), {
            "hasDriver": _iris("Driver", results["driverId"]),
            "hasTeam": _iris("Team", results["constructorId"]),
            "hasStartPosition": _positions(results["grid"]),
            "hasFinishPosition": _positions(results["position"]),
            "hasLapsCompleted": results["laps"],
            "hasRacePoints": _flags(results["points"]),
            "hasStatus": results["statusText"],
        }),
        ("RaceEntry", _iris("RaceEntry", quali["entryId"]), {"hasSessionResult": quali_iri}),
        ("RaceEntry", _iris("RaceEntry", sprint["entryId"]), {"hasSessionResult": sprint_iri}),
        ("RaceEntry", _iris("RaceEntry", practice["entryId"]), {"hasSessionResult": practice_iri}),
        ("SessionResult", quali_iri, {
            "hasSessionType": ["Qualifying"] * len(quali),
            "hasPosition": _positions(quali["position"]),
            "hasBestLapTime": [_decimal_str(best_lap(*q)) for q in zip(quali["q1"], quali["q2"], quali["q3"])],
        }),
        ("SessionResult", sprint_iri, {
            "hasSessionType": ["Sprint"] * len(sprint),
            "hasPosition": _positions(sprint["position"]),
            "hasBestLapTime": [_decimal_str(best_lap(t)) for t in sprint["fastestLapTime"]],
            "hasSessionLapsCompleted": sprint["laps"],
            "hasSessionPoints": _flags(sprint["points"]),
        }),
        ("SessionResult", practice_iri, {
            "hasSessionType": practice["session"],
            "hasPosition": _positions(practice["position"]),
            "hasBestLapTime": [_decimal_str(practice_lap_time_to_seconds(t)) for t in practice["bestLapTime"]],
            "hasSessionLapsCompleted": practice["laps"],
        }),
        ("Position", ranks.map(position_iri), {
            "hasRank": ranks.map(str),
        }),
    ]


def _assertions(subjects, cls: str, prop: str, values) -> pd.DataFrame:
    frame = pd.DataFrame({"subject": list(subjects), "subject_class": cls, "property": prop, "value": list(values)})
    return frame[frame["value"].notna()]


def build_assertions(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (entities, assertions):
      entities:   (iri, class) for every node, with the class's local name.
      assertions: (subject, subject_class, property, value), value being a literal's
                  lexical form or an object IRI.
    To reload the saved CSVs, read them with dtype=str, keep_default_na=False so lexical
    forms such as "true" and "78.050" survive unchanged.
    """
    sources = property_sources(_scope(tables))
    entities = pd.concat(
        [pd.DataFrame({"iri": list(subjects), "class": cls}) for cls, subjects, _ in sources]
    ).drop_duplicates(ignore_index=True)
    assertions = pd.concat(
        [_assertions(subjects, cls, prop, values)
         for cls, subjects, columns in sources for prop, values in columns.items()]
    ).drop_duplicates(ignore_index=True)
    return entities, assertions


def subset(entities: pd.DataFrame, assertions: pd.DataFrame, seeds,
           stop_properties=frozenset({"precedesRace", "precedesSeason", "hasTeamMember"})):
    """
    (entities, assertions) restricted to the seed nodes and every node reachable from them
    through object values. stop_properties are not followed, since they would chain to the
    whole graph; their triples are kept, pointing at nodes outside the subset.
    """
    iris = set(entities["iri"])
    nodes, frontier = set(), set(seeds)
    while frontier:
        nodes |= frontier
        step = assertions[assertions["subject"].isin(frontier) & ~assertions["property"].isin(stop_properties)]
        frontier = (set(step["value"]) & iris) - nodes
    return entities[entities["iri"].isin(nodes)], assertions[assertions["subject"].isin(nodes)]


# ── Rendering ─────────────────────────────────────────────────────────────

def _literal(value: str, datatype: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"^^<{datatype}>'


def render_ntriples(entities: pd.DataFrame, assertions: pd.DataFrame, schema: Schema) -> tuple[list[str], set[str]]:
    """
    Writes every assertion along its schema shape, plus rdf:type triples (the class and,
    if the schema aligns it, its direct SULO superclass). SULO path intermediate nodes
    get the IRI <subject>/<property> so the source property stays recoverable outside
    the model. Returns (sorted unique N-Triples lines, properties the schema does not declare).
    """
    lines = set()
    for node, cls in entities.itertuples(index=False):
        lines.add(f"<{node}> <{RDF_TYPE}> <{NS}{cls}> .")
        superclass = schema.sulo_superclass.get(cls)
        if superclass:
            lines.add(f"<{node}> <{RDF_TYPE}> <{superclass}> .")

    set_aside = set()
    for subject, cls, prop, value in assertions.itertuples(index=False):
        shape = schema.properties.get(prop)
        if shape is None:
            set_aside.add(prop)
            continue
        if shape.domain != NS + cls:
            raise ValueError(f"{prop}: schema domain {shape.domain} but subject class {cls}")
        obj = _literal(value, shape.datatype) if shape.datatype else f"<{value}>"
        if shape.hop is None:
            lines.add(f"<{subject}> <{shape.predicate}> {obj} .")
        else:
            hop_class, hop_predicate = shape.hop
            node = f"{subject}/{prop}"
            lines.add(f"<{subject}> <{shape.predicate}> <{node}> .")
            lines.add(f"<{node}> <{RDF_TYPE}> <{hop_class}> .")
            lines.add(f"<{node}> <{hop_predicate}> {obj} .")
    return sorted(lines), set_aside
