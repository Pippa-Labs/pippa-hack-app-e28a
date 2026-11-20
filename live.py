"""Parsing of live TfL responses. Pure: no network, no SDK.

Every function takes the decoded JSON exactly as TfL sent it and returns the
guide's own shapes, or raises LiveDataError when the shape is not what we
expect (a 200 that contains nothing is the usual failure: docs/API_APPS.md).
app.py catches LiveDataError and answers from the static router instead.

Requests mirror src/tfl.js, with the planner limited to the Tube and walking
(this app routes the Tube only; the fixtures were captured the same way).
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote, urlencode

import guide

BASE = "https://api.tfl.gov.uk"


class LiveDataError(Exception):
    """TfL answered, but not with something we can use."""


# ── Requests (URLs only; app.py does the fetching) ──────────────────────────

def planner_url(from_id: str, to_id: str, preference: str | None, modes: str = "tube,walking") -> str:
    query = {"mode": modes}
    if preference:
        query["journeyPreference"] = preference
    return f"{BASE}/Journey/JourneyResults/{quote(from_id, safe='')}/to/{quote(to_id, safe='')}?{urlencode(query)}"


def arrivals_url(stop_id: str) -> str:
    return f"{BASE}/StopPoint/{quote(stop_id, safe='')}/Arrivals"


def status_url(line_ids: list[str]) -> str:
    return f"{BASE}/Line/{','.join(quote(i, safe='') for i in line_ids)}/Status"


# ── Journeys (calls A, B, C) ────────────────────────────────────────────────

def _leg(raw: dict) -> guide.Leg:
    mode = (raw.get("mode") or {}).get("id")
    if mode not in ("tube", "walking"):
        raise LiveDataError(f"unexpected mode {mode!r}")
    dep, arr = raw.get("departurePoint") or {}, raw.get("arrivalPoint") or {}
    minutes = int(raw.get("duration") or 0)
    if mode == "walking":
        return guide.Leg("walking", minutes, board_name=guide.station_sign(dep.get("naptanId"), dep.get("commonName", "")),
                         alight_name=guide.station_sign(arr.get("naptanId"), arr.get("commonName", "")))
    lines = list(dict.fromkeys(o["lineIdentifier"]["id"] for o in raw.get("routeOptions") or []
                               if (o.get("lineIdentifier") or {}).get("id")))
    if not lines or not dep.get("naptanId") or not arr.get("naptanId"):
        raise LiveDataError("tube leg without a line or stations")
    stops = (raw.get("path") or {}).get("stopPoints") or []
    return guide.Leg(
        "tube", minutes, lines, guide.LINE_NAMES.get(lines[0], lines[0]),
        dep["naptanId"], arr["naptanId"],
        guide.station_sign(dep["naptanId"], dep.get("commonName", "")),
        guide.station_sign(arr["naptanId"], arr.get("commonName", "")),
        [s.get("id") for s in stops],
        [guide.station_sign(s.get("id"), s.get("name", "")) for s in stops],
    )


def parse_journeys(body) -> list[guide.Journey]:
    """A planner response -> journeys. [] when TfL found none."""
    if not isinstance(body, dict):
        raise LiveDataError("planner response is not an object")
    out = []
    for j in body.get("journeys") or []:
        legs = [_leg(l) for l in j.get("legs") or []]
        if not legs:
            continue
        out.append(guide.Journey(legs, int(j.get("duration") or 0), "live",
                                 j.get("startDateTime"), j.get("arrivalDateTime")))
    return out


def _wall(s: str | None):
    try:
        return datetime.fromisoformat((s or "")[:19])
    except ValueError:
        return None


def effective_minutes(*responses: list[guide.Journey]):
    """route.js: measure each journey from the earliest departure on offer,
    so a train 25 minutes later that is one minute shorter does not win."""
    starts = [t for js in responses for j in js if (t := _wall(j.start))]
    t0 = min(starts) if starts else None

    def eff(j: guide.Journey) -> float:
        end = _wall(j.arrival)
        return j.minutes if t0 is None or end is None else (end - t0).total_seconds() / 60
    return eff


def choose_live(a: list[guide.Journey], b: list[guide.Journey], c: list[guide.Journey]) -> guide.Choice | None:
    """route.js chooseRoute over planner answers A (fewest changes),
    B (fastest) and C (walk only)."""
    eff = effective_minutes(a, b)
    train = lambda js: [j for j in js if j.tube_legs]
    walk_only = lambda js: [j for j in js if j.legs and all(l.mode == "walking" for l in j.legs)]
    A = min(train(a), key=lambda j: (j.changes, eff(j)), default=None)
    B = min(train(b), key=lambda j: (eff(j), j.changes), default=None)
    walks = walk_only(c) or walk_only(a) + walk_only(b)
    W = min((j.minutes for j in walks), default=None)
    return guide.choose(A, B, W, effective=eff)


# ── Arrivals (call D) and status (call E) ───────────────────────────────────

PREDICTION_KEYS = ("id", "naptanId", "lineId", "lineName", "platformName", "direction",
                   "destinationNaptanId", "destinationName", "towards", "timeToStation", "modeName")


def parse_arrivals(body) -> list[dict]:
    if not isinstance(body, list):
        raise LiveDataError("arrivals response is not a list")
    out = []
    for p in body:
        if isinstance(p, dict) and p.get("naptanId") and p.get("lineId") and isinstance(p.get("timeToStation"), (int, float)):
            out.append({k: p.get(k) for k in PREDICTION_KEYS if p.get(k) is not None})
    return out


def parse_status(body) -> list[dict]:
    if not isinstance(body, list) or not body:
        raise LiveDataError("status response is empty")
    return [{"id": l.get("id"), "name": l.get("name"),
             "lineStatuses": [{"statusSeverityDescription": s.get("statusSeverityDescription")}
                              for s in l.get("lineStatuses") or []]}
            for l in body if isinstance(l, dict)]


def train_rows(board: dict) -> list[str]:
    """Up to 3 board rows as plain strings; destinations are exact signage."""
    if not board.get("live"):
        return []
    says = {"take": "take this one", "skip": "not this one", "unknown": "check the front of the train"}
    return [f'"{r["destination"]}" in {r["minutes"]} min: {says[r["mark"]]}' for r in board["rows"]]


def platform_word(board: dict) -> str | None:
    """'Westbound' when every ticked train leaves from the same direction sign."""
    if not board.get("live"):
        return None
    words = {guide.split_platform(p)["direction"] for p in board["platforms"]}
    words.discard(None)
    return words.pop() if len(words) == 1 else None


# ── The whole answer: static first, live on top when it works ───────────────

PLANNER_TIMEOUT = 5.0   # seconds; the three planner calls run together
BOARD_TIMEOUT = 4.0     # arrivals and status run together: about 9 s worst case
DATA_NOTE = "Powered by TfL Open Data."


def _live_choice(origin: guide.Place, dest: guide.Place, get_json) -> guide.Choice | None:
    from concurrent.futures import ThreadPoolExecutor

    a_id, b_id = origin.stations[0], dest.stations[0]
    urls = [planner_url(a_id, b_id, "leastinterchange"),
            planner_url(a_id, b_id, "leasttime"),
            planner_url(a_id, b_id, None, modes="walking")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(get_json, u, PLANNER_TIMEOUT) for u in urls]
        results = [parse_journeys(futures[0].result())]      # A must work
        for f in futures[1:]:
            try:                                             # B and C are extras (C 404s on long walks)
                results.append(parse_journeys(f.result()))
            except Exception:
                results.append([])
    choice = choose_live(*results)
    if choice is None:
        return None
    if not choice.walk_minutes:
        # TfL's walk-only call answers 404 for a long walk: use the estimate.
        choice.walk_minutes = guide.walk_minutes_between(origin.lat, origin.lon, dest.lat, dest.lon)
    if choice.primary is not None:
        # The planner went station to station; add a landmark's walks.
        choice.primary.minutes += origin.walk_to_station + dest.walk_to_station
        choice.tube_minutes = choice.primary.minutes
    else:
        choice.tube_minutes = guide.plan_static(origin, dest).tube_minutes
    return choice


def _board_and_status(choice: guide.Choice, get_json):
    from concurrent.futures import ThreadPoolExecutor

    first = choice.primary.tube_legs[0]
    leg = {"lines": first.lines, "boardId": first.board_id, "alightId": first.alight_id}
    line_ids = list(dict.fromkeys(i for l in choice.primary.tube_legs for i in l.lines))
    rows, words, status, takes = [], {}, None, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        arrivals_f = pool.submit(get_json, arrivals_url(first.board_id), BOARD_TIMEOUT)
        status_f = pool.submit(get_json, status_url(line_ids), BOARD_TIMEOUT)
        try:
            names = {s["id"]: s["name"] for s in guide.STATIONS}
            board = guide.build_board(parse_arrivals(arrivals_f.result()), leg, names=names)
            rows = train_rows(board)
            if (word := platform_word(board)):
                words[0] = word
            if board.get("live"):
                takes[0] = list(dict.fromkeys(r["destination"] for r in board["rows"] if r["mark"] == "take"))
        except Exception:
            pass
        try:
            status = guide.status_strip(parse_status(status_f.result()), line_ids)
        except Exception:
            status = None
    return rows, words, status, takes


def answer(origin: guide.Place, dest: guide.Place, get_json=None) -> tuple[dict, str]:
    """(data, narrative) for the tube.route outcome.

    `get_json(url, timeout)` fetches and decodes one TfL URL; app.py passes
    its network seam. Without it, or when any live call fails in any way
    (exception, timeout, non-200, an empty or odd body), the answer is the
    static one with uses_live_data False. A live failure never raises.
    """
    origin, dest = guide.best_access(origin, dest)
    choice = guide.plan_static(origin, dest)
    used_live, rows, words, status, takes = False, [], {}, None, {}
    if get_json is not None:
        try:
            live_choice = _live_choice(origin, dest, get_json)
        except Exception:
            live_choice = None
        if live_choice is not None:
            choice, used_live = live_choice, True
            if choice.kind == "tube":
                rows, words, status, takes = _board_and_status(choice, get_json)

    primary = choice.primary
    data = {
        "recommendation": "walk" if choice.kind == "walk" else "tube",
        "steps": guide.steps_for(choice, origin, dest, words, takes),
        "tube_minutes": int(choice.tube_minutes or 0),
        "walk_minutes": int(choice.walk_minutes),
        "changes": 0 if choice.kind == "walk" or primary is None else int(primary.changes),
        "next_trains": rows if choice.kind == "tube" else [],
        "service_status": guide.status_sentence(status),
        "uses_live_data": used_live,
        "data_note": DATA_NOTE,
    }
    return data, guide.narrative_for(choice, origin, dest)
