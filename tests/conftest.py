"""Fixtures shared by the F1 population and node-data tests: the real schemas and a
hand-built miniature of the Kaggle tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_population import SCHEMA_FILES, build_assertions, read_schema

ROOT = Path(__file__).resolve().parents[1]
NAN = np.nan


@pytest.fixture(scope="module")
def schemas():
    return {kg: read_schema(str(ROOT / path)) for kg, path in SCHEMA_FILES.items()}


def _frame(columns, rows):
    return pd.DataFrame(rows, columns=columns, dtype=object)


@pytest.fixture(scope="module")
def tables():
    return {
        "seasons": _frame(["year"], [["2009"], ["2010"], ["2011"], ["2013"]]),
        "races": _frame(["raceId", "year", "round", "circuitId", "name", "date"], [
            ["1", "2009", "1", "1", "Old GP", "2009-03-29"],
            ["2", "2009", "2", "3", "Indianapolis 500", "2009-05-30"],          # excluded
            ["3", "2009", "3", "1", "Later GP", "2009-06-14"],
            ["10", "2010", "1", "1", "Bahrain Grand Prix", "2010-03-14"],
            ["11", "2010", "2", "2", "Australian Grand Prix", "2010-03-28"],
            ["12", "2011", "1", "1", "Bahrain Grand Prix", "2011-03-13"],
            ["13", "2013", "1", "1", "Bahrain Grand Prix", "2013-04-21"],  # 2012 missing: season gap
        ]),
        "results": _frame(
            ["resultId", "raceId", "driverId", "constructorId", "grid", "position", "points", "laps", "statusId"], [
                ["100", "1", "1", "1", "1", "1", "10", "50", "1"],     # pre-2010, in scope
                ["105", "2", "4", "1", "1", "1", "8", "200", "1"],     # Indianapolis 500 -> excluded
                ["106", "3", "5", "1", "4", "3", "2", "50", "1"],      # shared drive -> both rows excluded
                ["107", "3", "5", "2", "5", "3", "2", "50", "1"],
                ["101", "10", "1", "1", "1", "1", "25", "50", "1"],    # started and finished P1
                ["102", "10", "2", "1", "0", NAN, "0", "10", "2"],     # pit-lane start, unclassified
                ["103", "11", "1", "1", "2", "1", "0.5", "30", "1"],   # fractional points
                ["104", "11", "2", "1", "3", "2", NAN, "30", "1"],     # points missing -> no label
            ]),
        "qualifying": _frame(["qualifyId", "raceId", "driverId", "position", "q1", "q2", "q3"], [
            ["500", "10", "1", "1", "1:30.000", "1:29.500", NAN],
            ["501", "10", "2", "2", NAN, NAN, NAN],
            ["502", "10", "3", "3", "1:31.000", NAN, NAN],             # no race entry -> dropped
        ]),
        "sprint_results": _frame(["resultId", "raceId", "driverId", "position", "points", "laps", "fastestLapTime"], [
            ["600", "11", "1", "1", "8", "20", "1:31.000"],
        ]),
        "practice_results": _frame(["raceId", "driverId", "session", "position", "bestLapTime", "laps"], [
            ["10", "1", "FP1", "2", "0 days 00:01:30.500000", "25"],
            ["10", "2", "FP1", "1", "0 days 00:01:31", "22"],
            ["10", "3", "FP2", "1", "0 days 00:01:29.000000", "20"],           # no race entry -> dropped
        ]),
        "drivers": _frame(["driverId", "forename", "surname", "dob", "nationality"], [
            ["1", "Max", "Verstappen", "1997-09-30", "Dutch"],
            ["2", 'Jean "JJ"', "Back\\slash", NAN, NAN],
            ["3", "Never", "Raced", "1990-01-01", "British"],
            ["4", "Indy", "Only", "1930-01-01", "American"],
            ["5", "Shared", "Drive", "1930-01-01", "Italian"],
        ]),
        "constructors": _frame(["constructorId", "name", "nationality"], [
            ["1", "Red Bull", "Austrian"],
            ["2", "Shared Car", "Italian"],                                    # only in the shared drive
        ]),
        "circuits": _frame(["circuitId", "name", "location", "country"], [
            ["1", "Bahrain International Circuit", "Sakhir", "Bahrain"],
            ["2", "Albert Park", "Mel\nbourne\r", "Australia"],
            ["3", "Indianapolis Motor Speedway", "Indianapolis", "USA"],       # only the Indy 500
        ]),
        "status": _frame(["statusId", "status"], [["1", "Finished"], ["2", "Engine"]]),
    }


@pytest.fixture(scope="module")
def built(tables):
    return build_assertions(tables)
