"""Tests through LocalRunner, the code the production runtime uses.

They import the Pippa SDK, which only the hackathon build installs, so they
run there and not on a laptop. test_guide.py covers the logic itself and runs
anywhere. No test touches the network: `_fetch` is replaced on the instance.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

if "pytest" not in sys.modules:
    # `python -m unittest` on a laptop: no SDK here, so skip this file.
    # Under pytest (the build) a missing SDK stays a hard failure.
    import unittest

    raise unittest.SkipTest("test_app.py needs the Pippa SDK, which only the build installs")

import pytest
from pippa_agentapp_sdk import LocalRunner, OutputShapeViolation

from app import App, MissingPlaces, SamePlace, UnknownPlace

HERE = Path(__file__).resolve().parent

# A fixed "now", so a test that passes today still passes next month.
NOW = datetime(2026, 11, 20, 14, 30, tzinfo=timezone.utc)

TYPES = {
    "recommendation": str, "steps": list, "tube_minutes": int, "walk_minutes": int,
    "changes": int, "next_trains": list, "service_status": str,
    "uses_live_data": bool, "data_note": str,
}


def _fixture(name: str) -> bytes:
    return (HERE / f"fixtures_{name}.json").read_bytes()


def _saved_tfl(key: str):
    """A stand-in for App._fetch that answers from saved TfL responses."""
    def fetch(url: str, timeout: float = 6.0) -> bytes:
        if "/Journey/JourneyResults/" in url:
            if "journeyPreference" not in url:
                body = _fixture(f"{key}_C")
                if b'"journeys"' not in body:
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                return body
            return _fixture(f"{key}_A" if "leastinterchange" in url else f"{key}_B")
        if "/Arrivals" in url:
            return _fixture("vic_nhg_arrivals") if "940GZZLUVIC" in url else b"[]"
        if "/Status" in url:
            return _fixture("status_all")
        raise AssertionError(f"unexpected url {url}")
    return fetch


def _blocked(url: str, timeout: float = 6.0) -> bytes:
    raise urllib.error.HTTPError(url, 403, "error code: 1010", {}, None)


def call(inputs: dict, *, fetch=_blocked, explicit: bool = False):
    app = App()
    app._fetch = fetch
    return LocalRunner(app).call(
        "tube.route", inputs, granted_secrets={}, now=NOW, request_id="test-1",
        explicit=explicit,
    ).result


def _check_shape(data: dict) -> None:
    assert set(data) == set(TYPES)
    for name, kind in TYPES.items():
        assert isinstance(data[name], kind), name
        if kind is int:
            assert not isinstance(data[name], bool), name
    assert all(isinstance(s, str) for s in data["steps"] + data["next_trains"])
    assert data["data_note"] == "Powered by TfL Open Data."


def test_the_manifest_is_valid():
    app = App()
    assert app.manifest.app_id.startswith("hack.")
    assert [o.id for o in app.manifest.outcomes] == ["tube.route"]


def test_every_field_with_live_tfl_working():
    result = call({"from_place": "Victoria", "to_place": "Notting Hill Gate"}, fetch=_saved_tfl("vic_nhg"))
    data = result.data
    _check_shape(data)
    assert data["uses_live_data"] is True
    assert (data["recommendation"], data["tube_minutes"], data["changes"]) == ("tube", 13, 0)
    assert 'Get off at "Notting Hill Gate", 5 stops later.' in data["steps"]
    assert len(data["next_trains"]) == 3
    assert '"Circle"' in result.narrative


def test_every_field_when_tfl_refuses():
    """Pippa's servers have been refused by TfL (403 1010): the static answer."""
    data = call({"from_place": "Victoria", "to_place": "Notting Hill Gate"}).data
    _check_shape(data)
    assert data["uses_live_data"] is False
    assert data["next_trains"] == []
    assert 'Get off at "Notting Hill Gate", 5 stops later.' in data["steps"]


@pytest.mark.parametrize("failure", [
    TimeoutError("timed out"),
    urllib.error.URLError("unreachable"),
    RuntimeError("anything"),
])
def test_any_fetch_failure_still_answers(failure):
    def fetch(url, timeout=6.0):
        raise failure
    data = call({"from_place": "South Kensington", "to_place": "Camden Town"}, fetch=fetch).data
    _check_shape(data)
    assert data["uses_live_data"] is False
    assert data["changes"] == 1
    assert 'Change at "Leicester Square".' in data["steps"]


def test_an_empty_200_still_answers():
    data = call({"from_place": "Piccadilly Circus", "to_place": "Heathrow Terminal 5"},
                fetch=lambda url, timeout=6.0: b"{}").data
    _check_shape(data)
    assert data["uses_live_data"] is False
    assert 'Get off at "Heathrow Terminal 5", the stop after "Heathrow Terminals 2 & 3".' in data["steps"]


def test_a_short_trip_says_walk():
    data = call({"from_place": "Leicester Square", "to_place": "Covent Garden"},
                fetch=_saved_tfl("lsq_cgn")).data
    _check_shape(data)
    assert data["recommendation"] == "walk"
    assert data["steps"] == ['Walk from "Leicester Square" to "Covent Garden". It takes about 6 minutes.']


def test_a_landmark_in_japanese():
    data = call({"from_place": "victoria station", "to_place": "大英博物館"}).data
    _check_shape(data)
    assert data["steps"][-1].startswith('Walk from "') and '"British Museum"' in data["steps"][-1]


def test_missing_places_say_what_to_type():
    for inputs in ({}, {"from_place": "Victoria"}, {"from_place": "", "to_place": "Bank"}):
        with pytest.raises(MissingPlaces) as failure:
            call(inputs)
        assert failure.value.user_message == (
            "Tell me where you are starting and where you are going, "
            "for example Victoria to Notting Hill Gate.")


def test_an_unknown_place_is_named():
    with pytest.raises(UnknownPlace) as failure:
        call({"from_place": "Narnia Central", "to_place": "Victoria"})
    assert failure.value.user_message == "I could not find a Tube station or landmark called Narnia Central."


def test_the_same_place_twice():
    with pytest.raises(SamePlace) as failure:
        call({"from_place": "Victoria", "to_place": "victoria station"})
    assert failure.value.user_message == "Those are the same station."


def test_picked_by_name_it_answers_and_refuses_as_usual():
    picked = call({"from_place": "Victoria", "to_place": "Notting Hill Gate"}, explicit=True)
    assert picked.data == call({"from_place": "Victoria", "to_place": "Notting Hill Gate"}).data
    with pytest.raises(MissingPlaces):
        call({"from_place": "", "to_place": ""}, explicit=True)


def test_the_runner_enforces_the_manifest(monkeypatch):
    from pippa_agentapp_sdk import Result

    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": "hello"}))
    with pytest.raises(OutputShapeViolation):
        call({"from_place": "Victoria", "to_place": "Bank"})


def test_fixtures_are_real_tfl_json():
    for path in HERE.glob("fixtures_*.json"):
        json.loads(path.read_text("utf-8"))
