"""Tube guide: pure logic. No Pippa SDK, no network.

Ported from the tubey-test web app (src/names.js, find.js, route.js,
board.js, cards.js, status.js) plus a static router of its own, so the app
can always answer with no network at all. Live TfL data, when it is
reachable, is parsed in live.py and passed in here.

"Match, don't read": every station name, line name, sign word and train
destination in the steps is in English, inside double quotes, exactly as
London shows it. Everything else is plain English that Pippa may translate.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    return json.loads(HERE.joinpath(name).read_text(encoding="utf-8"))


STATIONS: list[dict] = _load("stations.json")      # Tube stations only
ROUTES: dict[str, list[dict]] = _load("routes.json")  # {line: [{dir, via, stops}]}
LANDMARKS: list[dict] = _load("landmarks.json")   # 30 places, names in 11 languages

STATION_BY_ID = {s["id"]: s for s in STATIONS}

# ── Rules, as in the web app (config.example.js) ─────────────────────────────
SWITCH_MIN_SAVING = 8   # minutes: take the faster route only if it saves this much
WALK_MAX = 10           # minutes: walk if the walk is this short
WALK_MARGIN = 3         # minutes: or no more than this slower than the Tube
STOP_COUNT_MAX = 6      # count stops up to this; beyond, "the stop after X"
FUZZY_MIN = 0.62        # spelling closeness needed for a typo match
SAME_TRAIN_SECONDS = 90

# ── Static router (no network) ───────────────────────────────────────────────
MINUTES_PER_STOP = 2    # estimate: routes.json has stop order, not times
CHANGE_MINUTES = 5      # real time to change trains
CHANGE_PENALTY = 8      # extra preference against changing (not reported)
WAIT_MINUTES = 3
ACCESS_MINUTES = 4      # street to platform
WALK_DETOUR = 1.3       # streets are longer than a straight line
WALK_METRES_PER_MINUTE = 80

LINE_NAMES = {
    "bakerloo": "Bakerloo", "central": "Central", "circle": "Circle",
    "district": "District", "hammersmith-city": "Hammersmith & City",
    "jubilee": "Jubilee", "metropolitan": "Metropolitan", "northern": "Northern",
    "piccadilly": "Piccadilly", "victoria": "Victoria", "waterloo-city": "Waterloo & City",
}


class GuideError(Exception):
    """A question the guide cannot answer; app.py turns it into a user message."""


class MissingPlace(GuideError):
    pass


class UnknownPlace(GuideError):
    def __init__(self, text: str):
        super().__init__(f"no station or landmark matches {text!r}")
        self.text = text


class SamePlace(GuideError):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# Names (src/names.js)
# ═════════════════════════════════════════════════════════════════════════════

def clean_station_name(name: str = "") -> str:
    """'Battersea Power Station Underground Station' -> 'Battersea Power Station'."""
    return re.sub(r" (Underground |DLR |Rail )?Station$", "", name or "").strip()


def strip_bracket(name: str = "") -> str:
    """'Edgware Road (Circle Line)' -> 'Edgware Road'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name or "").strip()


def clean_destination(name: str = "") -> str:
    return strip_bracket(clean_station_name(name))


def fold_name(text: str = "") -> str:
    """Case-, accent- and punctuation-insensitive key, for every script.

    Latin accents are dropped, then text is recomposed so Japanese kana such
    as ガ stay whole (same as foldName in src/names.js).
    """
    s = unicodedata.normalize("NFD", text or "")
    s = "".join(ch for ch in s if not ("̀" <= ch <= "ͯ"))
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"['’.]", "", s).replace("&", " and ")
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


# Brackets that name a line are not on the station's sign:
# "Edgware Road (Circle Line)", "Shepherd's Bush (Central)". Others are:
# "Kensington (Olympia)".
_LINE_BRACKET = re.compile(r"\s*\((?:[^)]*\bLine|Central|Bakerloo)\)", re.I)


def sign_name(name: str) -> str:
    """What the station's own signs say, from a name in stations.json.

    Those names are already clean, so " Station" is NOT stripped here:
    "Battersea Power Station" is the station's name. (TfL planner names such
    as "Victoria Underground Station" go through clean_station_name first.)
    """
    name = re.sub(r"-Underground$", "", name or "")
    return _LINE_BRACKET.sub("", name).strip()


def station_sign(station_id: str, fallback: str = "") -> str:
    """Sign name for a station id; `fallback` is a raw TfL name for ids not in our data."""
    s = STATION_BY_ID.get(station_id)
    return sign_name(s["name"]) if s else sign_name(clean_station_name(fallback))


# Stations that are really one place (two Edgware Roads, two Hammersmiths,
# Paddington's two): a trip to or from the place may use either.
def _group_key(name: str) -> str:
    # "-Underground" first: "Paddington (H&C Line)-Underground" -> "Paddington".
    return fold_name(strip_bracket(re.sub(r"-Underground$", "", name)))


GROUPS: dict[str, list[str]] = {}
for _s in STATIONS:
    GROUPS.setdefault(_group_key(_s["name"]), []).append(_s["id"])


def station_group(station_id: str) -> list[str]:
    return GROUPS[_group_key(STATION_BY_ID[station_id]["name"])]


# ═════════════════════════════════════════════════════════════════════════════
# Places (src/find.js)
# ═════════════════════════════════════════════════════════════════════════════

def _bigrams(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    t = f" {s} "
    for i in range(len(t) - 1):
        g = t[i:i + 2]
        out[g] = out.get(g, 0) + 1
    return out


def similarity(a: str, b: str) -> float:
    """Dice coefficient over character pairs (same as find.js)."""
    if not a or not b:
        return 0.0
    A, B = _bigrams(a), _bigrams(b)
    common = sum(min(n, B.get(g, 0)) for g, n in A.items())
    total = sum(A.values()) + sum(B.values())
    return (2 * common) / total if total else 0.0


def _is_latin(s: str) -> bool:
    return all(ch == " " or ch.isdigit() or ("LATIN" in unicodedata.name(ch, "")) for ch in s)


@dataclass
class _Entry:
    kind: str           # "landmark" | "station"
    ref: dict
    keys: list[str]
    boost: int          # landmarks win ties over stations (find.js)


def _build_index() -> list[_Entry]:
    entries = []
    for p in LANDMARKS:
        keys = {fold_name(n) for n in p["names"].values()}
        for names in (p.get("aliases") or {}).values():
            keys |= {fold_name(n) for n in names}
        entries.append(_Entry("landmark", p, sorted(k for k in keys if k), 0))
    for s in STATIONS:
        keys = {fold_name(s["name"]), fold_name(sign_name(s["name"]))}
        entries.append(_Entry("station", s, sorted(keys), 1))
    return entries


INDEX = _build_index()

# "victoria station", "the british museum", "lester sq tube"
_NOISE = re.compile(r"^(the )|( (underground|tube|train|rail))?( station| stn)?$")


def _score(key: str, q: str):
    # One step finer than find.js: an exact name beats a name that merely
    # starts with the query, so "victoria" is Victoria station, not the V&A.
    if key == q:
        return -1
    if key.startswith(q):
        return 0
    if any(w.startswith(q) for w in key.split(" ")):
        return 1
    if q in key:
        return 2
    return None


EXACT_STATION_SCORE = -1 * 2 + 1   # _score -1 (exact), doubled, plus the station boost


def search(query: str, limit: int = 8) -> list[tuple[str, dict, float]]:
    """Best matches first: [(kind, ref, score)]."""
    q = fold_name(query)
    q = _NOISE.sub("", q).strip() or q
    if not q:
        return []
    hits = []
    for e in INDEX:
        best = None
        for k in e.keys:
            s = _score(k, q)
            if s is not None and (best is None or s < best):
                best = s
            if best == -1:
                break
        if best is None and len(q) >= 4 and _is_latin(q):
            sim = max(similarity(k, q) for k in e.keys)
            if sim >= FUZZY_MIN:
                best = 3 + (1 - sim)
        if best is not None:
            hits.append((e.kind, e.ref, best * 2 + e.boost))
    name = lambda h: h[1]["names"]["en"] if h[0] == "landmark" else h[1]["name"]
    hits.sort(key=lambda h: (h[2], name(h)))
    return hits[:limit]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = math.radians
    d_lat, d_lon = r(lat2 - lat1), r(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(d_lon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def walk_minutes_between(lat1, lon1, lat2, lon2) -> int:
    metres = haversine_m(lat1, lon1, lat2, lon2) * WALK_DETOUR
    return max(1, round_half_up(metres / WALK_METRES_PER_MINUTE))


def round_half_up(x: float) -> int:
    """JavaScript's Math.round (Python's round() rounds halves to even)."""
    return math.floor(x + 0.5)


@dataclass
class Place:
    name: str               # English, as on signs
    lat: float
    lon: float
    stations: list[str]     # station ids to route from/to (a group for twin stations)
    landmark: str | None = None
    walk_to_station: int = 0  # minutes from a landmark to its station
    # A landmark's nearby stations: [(station ids, walk minutes)], nearest first.
    options: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.landmark or ",".join(sorted(self.stations))


def nearest_station(lat: float, lon: float) -> dict:
    return min(STATIONS, key=lambda s: haversine_m(lat, lon, s["lat"], s["lon"]))


def resolve_place(text: str) -> Place:
    """A station, a landmark (in any app language) or a misspelling of one."""
    text = (text or "").strip()
    if not text:
        raise MissingPlace("empty place")
    hits = search(text, limit=8)
    if not hits:
        raise UnknownPlace(text)
    kind, ref, score = hits[0]
    # "Piccadilly Circus", "Covent Garden", "King's Cross St Pancras" are
    # landmarks and stations at once: an exact station name means the station.
    exact_station = next((h for h in hits if h[0] == "station" and h[2] == EXACT_STATION_SCORE), None)
    if kind == "landmark" and exact_station:
        kind, ref, score = exact_station
    if kind == "station":
        return Place(sign_name(ref["name"]), ref["lat"], ref["lon"], station_group(ref["id"]))
    options = nearby_stations(ref["lat"], ref["lon"])
    stations, walk = options[0]
    return Place(ref["names"]["en"], ref["lat"], ref["lon"], stations, landmark=ref["id"],
                 walk_to_station=walk, options=options)


LANDMARK_RADIUS_M = 1200
FAR_WALK = 20   # minutes: beyond this, say the Tube does not go close
LANDMARK_OPTIONS = 3


def nearby_stations(lat: float, lon: float) -> list[tuple[list[str], int]]:
    """Up to 3 stations within a short walk of a landmark (always at least the
    nearest), as (station group, walk minutes), nearest first."""
    ranked = sorted(STATIONS, key=lambda s: haversine_m(lat, lon, s["lat"], s["lon"]))
    out, seen = [], set()
    for st in ranked:
        group = tuple(station_group(st["id"]))
        if group in seen:
            continue
        if out and (len(out) >= LANDMARK_OPTIONS or haversine_m(lat, lon, st["lat"], st["lon"]) > LANDMARK_RADIUS_M):
            break
        seen.add(group)
        out.append((list(group), walk_minutes_between(lat, lon, st["lat"], st["lon"])))
    return out


def best_access(origin: Place, dest: Place) -> tuple[Place, Place]:
    """For a landmark, use whichever nearby station gives the quickest trip
    door to door (walk included), not simply the nearest one."""
    from dataclasses import replace
    o_opts = origin.options or [(origin.stations, origin.walk_to_station)]
    d_opts = dest.options or [(dest.stations, dest.walk_to_station)]
    if len(o_opts) == 1 and len(d_opts) == 1:
        return origin, dest
    best = None
    for o_st, o_walk in o_opts:
        for d_st, d_walk in d_opts:
            if set(o_st) & set(d_st):
                continue
            j = route_static(o_st, d_st, CHANGE_PENALTY)
            if j is None:
                continue
            total = j.minutes + o_walk + d_walk + CHANGE_PENALTY * j.changes
            if best is None or total < best[0]:
                best = (total, o_st, o_walk, d_st, d_walk)
    if best is None:
        return origin, dest
    _, o_st, o_walk, d_st, d_walk = best
    return (replace(origin, stations=o_st, walk_to_station=o_walk),
            replace(dest, stations=d_st, walk_to_station=d_walk))


# ═════════════════════════════════════════════════════════════════════════════
# Journeys: one shape for the static router and for live TfL answers
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Leg:
    mode: str                     # "tube" | "walking"
    minutes: int
    lines: list[str] = field(default_factory=list)
    line_name: str = ""
    board_id: str = ""
    alight_id: str = ""
    board_name: str = ""          # as on signs
    alight_name: str = ""
    stop_ids: list[str] = field(default_factory=list)    # after boarding, alighting stop last
    stop_names: list[str] = field(default_factory=list)


@dataclass
class Journey:
    legs: list[Leg]
    minutes: int                  # real minutes, door to door
    source: str                   # "static" | "live"
    start: str | None = None      # live: planner start time, "YYYY-MM-DDTHH:MM:SS"
    arrival: str | None = None

    @property
    def tube_legs(self) -> list[Leg]:
        return [l for l in self.legs if l.mode == "tube"]

    @property
    def changes(self) -> int:
        return max(0, len(self.tube_legs) - 1)

    def signature(self) -> str:
        return "|".join(f"{'+'.join(sorted(l.lines))}:{l.board_id}>{l.alight_id}" for l in self.tube_legs)


# ═════════════════════════════════════════════════════════════════════════════
# Static router: Dijkstra over (station, line, branch) nodes
# ═════════════════════════════════════════════════════════════════════════════
#
# One node per branch, not per line: staying on a branch is staying on one
# train, and moving to another branch (or line) is a change of train. Found
# while testing: with one node per line, Leicester Square -> Camden Town rode
# the Charing Cross branch to Euston and slipped onto the Bank branch there,
# a change of platform that cost nothing and skipped Mornington Crescent.

# Stops in TfL's route data that the line's trains run through for most of
# the day. Piccadilly trains call at Turnham Green only early and late, and
# TfL's own Piccadilly Circus -> Heathrow journey lists 19 stops without it;
# leaving it in would make a visitor's stop count one off.
DAYTIME_SKIPS = {"piccadilly": {"940GZZLUTNG"}}  # Turnham Green


def _build_graph():
    ride: dict[tuple, set] = {}
    at_station: dict[str, set] = {}
    for line, branches in ROUTES.items():
        for bi, b in enumerate(branches):
            stops = [st for st in b["stops"] if st not in DAYTIME_SKIPS.get(line, ())]
            for a, c in zip(stops, stops[1:]):
                if a != c:
                    ride.setdefault((a, line, bi), set()).add((c, line, bi))
            for st in stops:
                at_station.setdefault(st, set()).add((st, line, bi))
    return ride, at_station


RIDE, AT_STATION = _build_graph()


def route_static(start_ids: list[str], end_ids: list[str], change_penalty: int) -> Journey | None:
    """Best Tube route between two sets of stations, or None.

    Cost = real minutes + change_penalty per change. Real minutes = wait and
    platform access once, 2 per stop, 5 per change.
    """
    ends = set(end_ids)
    start_cost = WAIT_MINUTES + ACCESS_MINUTES
    dist: dict[tuple, float] = {}
    prev: dict[tuple, tuple | None] = {}
    heap: list = []
    for st in start_ids:
        for node in AT_STATION.get(st, ()):
            dist[node] = start_cost
            prev[node] = None
            heapq.heappush(heap, (start_cost, node))
    goal = None
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        if node[0] in ends:
            goal = node
            break
        steps = [(n, MINUTES_PER_STOP) for n in RIDE.get(node, ())]
        steps += [(n, CHANGE_MINUTES + change_penalty) for n in AT_STATION.get(node[0], ()) if n != node]
        for nxt, w in steps:
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    if goal is None:
        return None

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    legs: list[Leg] = []
    current: Leg | None = None
    for a, b in zip(path, path[1:]):
        if a[0] == b[0]:            # change of line or direction at a station
            current = None
            continue
        if current is None:
            current = Leg("tube", 0, [a[1]], LINE_NAMES.get(a[1], a[1]), a[0], b[0], station_sign(a[0]))
            legs.append(current)
        current.stop_ids.append(b[0])
        current.alight_id = b[0]
    for leg in legs:
        leg.minutes = MINUTES_PER_STOP * len(leg.stop_ids)
        leg.alight_name = station_sign(leg.alight_id)
        leg.stop_names = [station_sign(i) for i in leg.stop_ids]
    if not legs:
        return None
    minutes = start_cost + sum(l.minutes for l in legs) + CHANGE_MINUTES * (len(legs) - 1)
    return Journey(legs, minutes, "static")


# ═════════════════════════════════════════════════════════════════════════════
# Route choice (src/route.js)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Choice:
    kind: str                        # "walk" | "tube"
    primary: Journey | None          # the Tube journey (also kept for a walk answer)
    walk_minutes: int
    tube_minutes: int | None
    reason: dict | None = None       # {"saved": n, "extra": m} when the faster route won
    alternative: Journey | None = None


def choose(fewest: Journey | None, fastest: Journey | None, walk_minutes: int | None,
           effective=None) -> Choice | None:
    """route.js chooseRoute: fewest changes by default, the faster route only if
    it saves SWITCH_MIN_SAVING minutes, and walk when that is short or close.

    `effective(j)` gives the minutes to compare (live: from the earliest
    departure on offer, so a later but shorter train does not win)."""
    eff = effective or (lambda j: j.minutes)
    A, B = fewest or fastest, fastest or fewest
    W = math.inf if walk_minutes is None else walk_minutes
    if A is None:
        return Choice("walk", None, int(W), None) if W != math.inf else None
    if W <= WALK_MAX or W <= eff(A) + WALK_MARGIN:
        return Choice("walk", A, int(W), A.minutes, alternative=A)
    saved = round_half_up(eff(A) - eff(B))
    if saved >= SWITCH_MIN_SAVING and B.changes > A.changes:
        return Choice("tube", B, int(W) if W != math.inf else 0, B.minutes,
                      reason={"saved": saved, "extra": B.changes - A.changes}, alternative=A)
    alt = None if A.signature() == B.signature() else B
    return Choice("tube", A, int(W) if W != math.inf else 0, A.minutes, alternative=alt)


def plan_static(origin: Place, dest: Place) -> Choice:
    """The answer with no network at all."""
    walk = walk_minutes_between(origin.lat, origin.lon, dest.lat, dest.lon)
    if set(origin.stations) & set(dest.stations):
        # Different places that share a station (two landmarks near one stop):
        # there is nothing to ride.
        return Choice("walk", None, walk, None)
    access = origin.walk_to_station + dest.walk_to_station
    fewest = route_static(origin.stations, dest.stations, CHANGE_PENALTY)
    fastest = route_static(origin.stations, dest.stations, 0)
    for j in (fewest, fastest):
        if j:
            j.minutes += access
    return choose(fewest, fastest, walk) or Choice("walk", None, walk, None)


# ═════════════════════════════════════════════════════════════════════════════
# Which trains to board (src/board.js)
# ═════════════════════════════════════════════════════════════════════════════

HR4, HRC, HNX = "940GZZLUHR4", "940GZZLUHRC", "940GZZLUHNX"

# Metropolitan fast and semi-fast trains skip these, and arrivals do not say
# which trains are fast. Never tick a Metropolitan train to one.
MET_FAST_SKIPS = {
    "940GZZLUWYP",  # Wembley Park
    "940GZZLUPRD",  # Preston Road
    "940GZZLUNKP",  # Northwick Park
    "940GZZLUNHA",  # North Harrow
    "940GZZLUPNR",  # Pinner
    "940GZZLUNWH",  # Northwood Hills
    "940GZZLUNOW",  # Northwood
}

VIA_ALIASES = {"cx": "charing cross"}


def effective_branches(line_id: str, branches: list[dict]) -> list[dict]:
    """Heathrow loop: trains shown as "Heathrow Terminal 4" run Hatton Cross ->
    T4 -> T2&3, but TfL's sequence stops at T4."""
    if line_id != "piccadilly":
        return branches
    out = []
    for b in branches:
        s = b["stops"]
        if len(s) >= 2 and s[-1] == HR4 and s[-2] == HNX:
            b = {**b, "stops": s + [HRC]}
        out.append(b)
    return out


def loop_train(line_id: str, dest_id: str, branches: list[dict]):
    """A T4-loop train is judged only on the loop branch, ending at T2&3."""
    if line_id == "piccadilly" and dest_id == HR4:
        return HRC, [b for b in branches if HR4 in b["stops"]]
    return dest_id, branches


def via_of(towards: str | None) -> str | None:
    m = re.search(r"\bvia\s+(.+)$", towards or "", re.I)
    if not m:
        return None
    v = fold_name(m.group(1))
    return VIA_ALIASES.get(v, v)


def matcher(aliases: dict | None = None):
    aliases = aliases or {}
    return lambda a, b: a == b or aliases.get(a) == b or aliases.get(b) == a or (
        bool(aliases.get(a)) and aliases.get(a) == aliases.get(b))


def _find(stops, ident, start, same) -> int:
    for i in range(start, len(stops)):
        if same(stops[i], ident):
            return i
    return -1


def verdicts(stops, board, alight, dest, same=None) -> list[bool]:
    """For each time `board` is on the branch: does a train ending at the first
    `dest` after it call at `alight` on the way? (The Circle repeats stops.)"""
    same = same or (lambda a, b: a == b)
    out = []
    for i, s in enumerate(stops):
        if not same(s, board):
            continue
        k = _find(stops, dest, i + 1, same)
        if k < 0:
            continue
        j = _find(stops, alight, i + 1, same)
        out.append(i < j <= k)
    return out


def judge(branches, board, alight, dest, same=None) -> str:
    v = [x for b in branches for x in verdicts(b["stops"], board, alight, dest, same)]
    if not v:
        return "none"
    if all(v):
        return "take"
    if not any(v):
        return "skip"
    return "mixed"


def destination_by_name(p: dict, routes: dict, aliases: dict, stop_names: dict):
    want = fold_name(p.get("destinationName") or "")
    if not want:
        return None
    branches = routes.get(p.get("lineId"), [])
    if p.get("direction"):
        branches = [b for b in branches if b["dir"] == p["direction"]]
    hits = {aliases.get(i, i) for b in branches for i in b["stops"] if fold_name(stop_names.get(i, "")) == want}
    return next(iter(hits)) if len(hits) == 1 else None


def classify(p: dict, leg: dict, routes: dict = ROUTES, aliases: dict | None = None,
             stop_names: dict | None = None) -> str:
    """"take" only when the data proves the train calls at the alighting stop;
    "skip" when it proves it does not; otherwise "unknown" (no mark).

    `leg` is {"lines": [...], "boardId": ..., "alightId": ...}, the shape
    src/main.js builds.
    """
    aliases, stop_names = aliases or {}, stop_names or {}
    dest_raw = p.get("destinationNaptanId") or destination_by_name(p, routes, aliases, stop_names)
    if not dest_raw:
        return "unknown"
    same = matcher(aliases)
    has = lambda b, i: any(same(s, i) for s in b["stops"])
    line = effective_branches(p.get("lineId"), routes.get(p.get("lineId"), []))
    if not any(has(b, leg["boardId"]) for b in line):
        return "unknown"
    if p.get("lineId") in leg["lines"] and not any(has(b, leg["alightId"]) for b in line):
        return "unknown"
    dest, every = loop_train(p.get("lineId"), dest_raw, line)
    if not any(has(b, dest) for b in every):
        return "unknown"

    branches = every
    if p.get("direction"):
        branches = [b for b in branches if b["dir"] == p["direction"]]
    via = via_of(p.get("towards"))
    verdict = "none"
    if via:
        verdict = judge([b for b in branches if b.get("via") and fold_name(b["via"]) == via],
                        leg["boardId"], leg["alightId"], dest, same)
        if verdict == "none":
            branches = [b for b in branches if not b.get("via")]
    if verdict == "none":
        verdict = judge(branches, leg["boardId"], leg["alightId"], dest, same)
    if verdict == "none":
        return "skip" if judge(every, leg["boardId"], leg["alightId"], dest, same) == "none" else "unknown"
    if verdict == "mixed":
        return "unknown"
    if verdict == "take":
        if p.get("lineId") not in leg["lines"]:
            return "unknown"
        if p.get("lineId") == "metropolitan" and leg["alightId"] in MET_FAST_SKIPS:
            return "unknown"
        return "take"
    return "skip"


def dedupe(predictions: list[dict]) -> list[dict]:
    """One train listed several times by TfL shows once."""
    ids, kept = set(), []
    key = lambda p: f"{p.get('lineId')}|{(p.get('platformName') or '').strip()}|{p.get('destinationNaptanId') or p.get('destinationName')}"
    for p in sorted(predictions, key=lambda x: x.get("timeToStation") or 0):
        if p.get("id") and p["id"] in ids:
            continue
        if any(key(k) == key(p) and abs((k.get("timeToStation") or 0) - (p.get("timeToStation") or 0)) < SAME_TRAIN_SECONDS
               for k in kept):
            continue
        if p.get("id"):
            ids.add(p["id"])
        kept.append(p)
    return kept


def minutes_of(seconds) -> int:
    return max(1, round_half_up((seconds or 0) / 60))


def display_destination(p: dict) -> str:
    if p.get("modeName") == "bus":
        return (p.get("destinationName") or "").strip()
    return (p.get("towards") or "").strip() or clean_destination(p.get("destinationName") or "")


def static_termini(leg: dict, routes: dict = ROUTES, names: dict | None = None, aliases=None, stop_names=None) -> list[str]:
    names, stop_names = names or {}, stop_names or {}
    same = matcher(aliases)
    seen: list[str] = []
    for line_id in leg["lines"]:
        for b in effective_branches(line_id, routes.get(line_id, [])):
            if not any(verdicts(b["stops"], leg["boardId"], leg["alightId"], b["stops"][-1], same)):
                continue
            terminus = b["stops"][-1]
            if line_id == "piccadilly" and terminus == HRC and b["stops"][-2] == HR4:
                terminus = HR4
            name = names.get(terminus) or stop_names.get(terminus)
            if not name:
                continue
            label = strip_bracket(name) + (f" via {b['via']}" if b.get("via") else "")
            if label not in seen:
                seen.append(label)
    return seen


def build_board(arrivals: list[dict], leg: dict, routes: dict = ROUTES, rows: int = 3,
                names: dict | None = None, aliases=None, stop_names=None) -> dict:
    """The live board for one leg, same shape as buildBoard in board.js."""
    here = [{**p, "platformName": (p.get("platformName") or "").strip(),
             "mark": classify(p, leg, routes, aliases, stop_names)}
            for p in dedupe([p for p in arrivals or [] if p.get("naptanId") == leg["boardId"]])]
    here.sort(key=lambda p: p.get("timeToStation") or 0)
    takes = [p for p in here if p["mark"] == "take"]
    if not takes:
        return {"live": False, "termini": static_termini(leg, routes, names, aliases, stop_names)}
    platforms = list(dict.fromkeys(p["platformName"] for p in takes))
    on_platforms = [p for p in here if p["platformName"] in platforms]
    shown = on_platforms[:rows]
    if not any(p["mark"] == "take" for p in shown):
        shown = shown[:rows - 1] + [takes[0]]
    out = [{
        "destination": display_destination(p),
        "minutes": minutes_of(p.get("timeToStation")),
        "mark": p["mark"],
        "lineId": p.get("lineId"),
        "lineName": p.get("lineName") or p.get("lineId"),
        "mode": p.get("modeName"),
        "platformName": p["platformName"],
    } for p in shown]
    index = next(i for i, r in enumerate(out) if r["mark"] == "take")
    return {"live": True, "rows": out, "platforms": platforms,
            "firstTake": {"index": index, "minutes": out[index]["minutes"]}}


def split_platform(platform_name: str = "") -> dict:
    """'Westbound - Platform 1' -> {'direction': 'Westbound', 'platform': 'Platform 1'}."""
    s = (platform_name or "").strip()
    tidy = lambda p: f"Platform {p}" if re.fullmatch(r"[A-Z0-9]{1,3}", p, re.I) else p
    i = s.find(" - ")
    if i < 0:
        return {"direction": None, "platform": tidy(s) if s else None}
    return {"direction": s[:i].strip(), "platform": tidy(s[i + 3:].strip())}


# ── Signs to look for on the platform (static, from routes.json) ─────────────

def boarding_signs(leg: Leg, routes: dict = ROUTES) -> tuple[list[str], list[str]]:
    """Destinations to board, and wrong ones that use the same platform.

    A train is on your platform when it is your line going your direction
    (the direction of the branches that run from your stop to the stop you
    get off at), or another line whose next stop is your next stop (the
    District beside the Circle at Victoria). Board the ones that reach your
    stop; the rest are the ones to avoid. A "via" is shown only when it is
    what tells two trains to the same terminus apart (Northern line).
    """
    board, alight = leg.board_id, leg.alight_id
    own = leg.lines[0] if leg.lines else None
    nxt = leg.stop_ids[0] if leg.stop_ids else None
    own_dirs = {b["dir"] for b in effective_branches(own, routes.get(own, []))
                if any(verdicts(b["stops"], board, alight, alight))}
    good: list[tuple[str, str | None]] = []
    bad: list[tuple[str, str | None]] = []
    for line_id, branches in routes.items():
        for b in effective_branches(line_id, branches):
            s = b["stops"]
            for i in range(len(s) - 1):
                if s[i] != board:
                    continue
                on_platform = b["dir"] in own_dirs if line_id == own else s[i + 1] == nxt
                if not on_platform:
                    continue
                terminus = s[-1]
                if line_id == "piccadilly" and terminus == HRC and s[-2] == HR4:
                    terminus = HR4                      # shown as "Heathrow Terminal 4"
                reaches = alight in s[i + 1:]
                if reaches and line_id == own:
                    good.append((terminus, b.get("via")))
                elif not reaches:
                    bad.append((terminus, b.get("via")))

    def label(terminus: str, via: str | None, others) -> str:
        name = station_sign(terminus)
        needs_via = via and any(t == terminus and v != via for t, v in others)
        return f"{name} via {via}" if needs_via else name

    show = list(dict.fromkeys(label(t, v, bad) for t, v in good))
    avoid = list(dict.fromkeys(label(t, v, good) for t, v in bad))
    return show, [a for a in avoid if a not in show]


# ═════════════════════════════════════════════════════════════════════════════
# Leaving the train (src/cards.js)
# ═════════════════════════════════════════════════════════════════════════════

def leave_text(leg: Leg) -> str:
    stops = len(leg.stop_ids) or len(leg.stop_names)
    if stops and stops <= STOP_COUNT_MAX:
        return f'Get off at "{leg.alight_name}", {stops} stop{"s" if stops != 1 else ""} later.'
    if len(leg.stop_names) >= 2:
        return f'Get off at "{leg.alight_name}", the stop after "{leg.stop_names[-2]}".'
    return f'Get off at "{leg.alight_name}".'


# ═════════════════════════════════════════════════════════════════════════════
# Line status (src/status.js)
# ═════════════════════════════════════════════════════════════════════════════

_CLOSED = re.compile(r"closure|closed|suspended", re.I)


def status_strip(status_response: list[dict] | None, line_ids: list[str]) -> dict:
    wanted = set(line_ids)
    lines = [l for l in status_response or [] if l.get("id") in wanted]
    descriptions = list(dict.fromkeys(s.get("statusSeverityDescription") for l in lines
                                      for s in l.get("lineStatuses") or [] if s.get("statusSeverityDescription")))
    per_line = [{"line": l.get("name") or l.get("id"), "description": s["statusSeverityDescription"]}
                for l in lines for s in l.get("lineStatuses") or []
                if s.get("statusSeverityDescription") and s["statusSeverityDescription"] != "Good Service"]
    per_line = list({f"{x['line']}|{x['description']}": x for x in per_line}.values())
    worst_first = lambda xs: [x for x in xs if _CLOSED.search(x["description"])] + [x for x in xs if not _CLOSED.search(x["description"])]
    if not descriptions:
        return {"state": None, "descriptions": [], "problems": []}
    if all(d == "Good Service" for d in descriptions):
        return {"state": "ok", "descriptions": descriptions, "problems": []}
    problems = [d for d in descriptions if d != "Good Service"]
    if any(_CLOSED.search(d) for d in problems):
        sorted_ = [d for d in problems if _CLOSED.search(d)] + [d for d in problems if not _CLOSED.search(d)]
        return {"state": "closed", "descriptions": sorted_, "problems": worst_first(per_line)}
    return {"state": "delay", "descriptions": problems, "problems": worst_first(per_line)}


def status_sentence(strip: dict | None) -> str:
    if not strip or not strip.get("state"):
        return "Not checked: live TfL status was unavailable."
    if strip["state"] == "ok":
        return '"Good Service" on every line in this journey.'
    worst = "; ".join(f'"{p["line"]}": "{p["description"]}"' for p in strip["problems"])
    advice = "Part of a line is closed: ask station staff." if strip["state"] == "closed" else "Allow extra time."
    return f"{worst}. {advice}"


# ═════════════════════════════════════════════════════════════════════════════
# The answer
# ═════════════════════════════════════════════════════════════════════════════

def minutes_text(n: int) -> str:
    return f"{n} minute" if n == 1 else f"{n} minutes"


def _or_list(names: list[str]) -> str:
    quoted = [f'"{n}"' for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def steps_for(choice: Choice, origin: Place, dest: Place, platform_words: dict[int, str] | None = None,
              live_takes: dict[int, list[str]] | None = None) -> list[str]:
    """Short English imperatives; text in double quotes is exact signage."""
    platform_words = platform_words or {}
    if choice.kind == "walk":
        return [f'Walk from "{origin.name}" to "{dest.name}". It takes about {minutes_text(choice.walk_minutes)}.']
    j = choice.primary
    steps: list[str] = []
    tube = j.tube_legs
    if origin.landmark and tube:
        if origin.walk_to_station > FAR_WALK:
            steps.append(f'The Tube does not go close to "{origin.name}". "{tube[0].board_name}" is about '
                         f'{minutes_text(origin.walk_to_station)} away on foot: ask about the "DLR" or a bus.')
        else:
            steps.append(f'Walk from "{origin.name}" to "{tube[0].board_name}" station. '
                         f'It takes about {minutes_text(origin.walk_to_station)}.')
    for n, leg in enumerate(j.legs):
        if leg.mode != "tube":
            if 0 < n < len(j.legs) - 1:
                steps.append(f'Walk to "{j.legs[n + 1].board_name}". It takes about {minutes_text(leg.minutes)}.')
            continue
        t = tube.index(leg)
        if t > 0:
            steps.append(f'Change at "{leg.board_name}".')
        line = f'At "{leg.board_name}", follow signs for the "{leg.line_name}" line'
        if t in platform_words:
            line += f', platform sign "{platform_words[t]}"'
        steps.append(line + ".")
        show, avoid = boarding_signs(leg)
        # Trains the live board proved right (a short working such as
        # "Arnos Grove") belong in the list too, exactly as their sign reads.
        show += [d for d in (live_takes or {}).get(t, []) if d not in show and clean_destination(d) not in show]
        if show:
            steps.append(f"Board a train showing {_or_list(show)}.")
        if avoid:
            steps.append(f"Do not board trains showing {_or_list(avoid)}.")
        steps.append(leave_text(leg))
    if dest.landmark and tube:
        if dest.walk_to_station > FAR_WALK:
            # Greenwich, London City Airport: the DLR goes there, the Tube does not.
            steps.append(f'The Tube does not go close to "{dest.name}". From "{tube[-1].alight_name}" it is '
                         f'about {minutes_text(dest.walk_to_station)} on foot: ask station staff about the "DLR" or a bus.')
        else:
            steps.append(f'Walk from "{tube[-1].alight_name}" to "{dest.name}". '
                         f'It takes about {minutes_text(dest.walk_to_station)}.')
    return steps


def narrative_for(choice: Choice, origin: Place, dest: Place) -> str:
    if choice.kind == "walk":
        quicker = " It is quicker than the Tube." if choice.tube_minutes else ""
        return f'Walk from "{origin.name}" to "{dest.name}": about {minutes_text(choice.walk_minutes)}.{quicker}'
    j = choice.primary
    tube = j.tube_legs
    parts = [f'the "{tube[0].line_name}" line from "{tube[0].board_name}" to "{tube[0].alight_name}"']
    parts += [f'then the "{l.line_name}" line to "{l.alight_name}"' for l in tube[1:]]
    changes = "no changes" if j.changes == 0 else f"{j.changes} change{'s' if j.changes > 1 else ''}"
    estimate = "" if j.source == "live" else " (an offline estimate)"
    end = ""
    if dest.landmark:
        end = (f', then about {minutes_text(dest.walk_to_station)} on foot or the "DLR" to "{dest.name}"'
               if dest.walk_to_station > FAR_WALK else f', then walk to "{dest.name}"')
    return f"Take {', '.join(parts)}{end}: about {minutes_text(j.minutes)}{estimate}, with {changes}."
