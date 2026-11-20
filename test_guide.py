"""Tests for guide.py and live.py. Standard library only (unittest), so they run
here (`python -m unittest`) and in the hackathon build (pytest collects them).

No test touches the network: live answers come from saved TfL responses
(fixtures_*.json), and the expected values in fixtures_expected_js.json were
produced by running the web app's own JavaScript on those same files.
"""

from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from pathlib import Path

import guide
import live

HERE = Path(__file__).resolve().parent
TRUTH = json.loads((HERE / "fixtures_expected_js.json").read_text("utf-8"))
NAMES = {s["id"]: s["name"] for s in guide.STATIONS}

JOURNEYS = {
    "vic_nhg": ("Victoria", "Notting Hill Gate"),
    "pcc_hr5": ("Piccadilly Circus", "Heathrow Terminal 5"),
    "lsq_cgn": ("Leicester Square", "Covent Garden"),
    "sks_ctn": ("South Kensington", "Camden Town"),
}


def fixture(name):
    return json.loads((HERE / f"fixtures_{name}.json").read_text("utf-8"))


def saved_tfl(key):
    """A get_json that answers from the saved responses for one journey."""
    arrivals = {"940GZZLUVIC": "vic_nhg_arrivals", "940GZZLUPCC": "pcc_hr5_arrivals",
                "940GZZLUSKS": "sks_ctn_arrivals", "940GZZLUCTN": "ctn_arrivals"}

    def get_json(url, timeout):
        assert timeout <= 6, "every request has a short timeout"
        if "/Journey/JourneyResults/" in url:
            if "mode=walking" in url and "journeyPreference" not in url:
                body = fixture(f"{key}_C")
                if "journeys" not in body:      # TfL answers 404 for a long walk
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                return body
            return fixture(f"{key}_A" if "leastinterchange" in url else f"{key}_B")
        if "/Arrivals" in url:
            stop = url.split("/StopPoint/")[1].split("/")[0]
            return fixture(arrivals[stop]) if stop in arrivals else []
        if "/Status" in url:
            return fixture("status_all")
        raise AssertionError(f"unexpected url {url}")
    return get_json


class NoNetwork(unittest.TestCase):
    """Like the template's conftest.py: any real connection fails the test."""

    def setUp(self):
        self._socket, self._create = socket.socket, socket.create_connection

        class Blocked(socket.socket):
            def connect(self, *a):
                raise AssertionError("a test tried to use the network")
            connect_ex = connect
        socket.socket = Blocked
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network"))

    def tearDown(self):
        socket.socket, socket.create_connection = self._socket, self._create


def run(key, get_json):
    a, b = JOURNEYS[key]
    return live.answer(guide.resolve_place(a), guide.resolve_place(b), get_json)


# ── The four journeys, live path (saved TfL answers) ────────────────────────

class LiveJourneys(NoNetwork):
    def test_victoria_to_notting_hill_gate(self):
        data, story = run("vic_nhg", saved_tfl("vic_nhg"))
        self.assertEqual(data["recommendation"], "tube")
        self.assertEqual((data["tube_minutes"], data["changes"]), (13, 0))
        self.assertIn('At "Victoria", follow signs for the "Circle" line, platform sign "Westbound".', data["steps"])
        self.assertIn('Get off at "Notting Hill Gate", 5 stops later.', data["steps"])
        self.assertTrue(data["uses_live_data"])
        self.assertIn('"Circle"', story)
        self.assertGreater(data["walk_minutes"], 30, "TfL 404s the long walk: the estimate is used, never 0")

    def test_piccadilly_circus_to_heathrow_t5(self):
        data, _ = run("pcc_hr5", saved_tfl("pcc_hr5"))
        self.assertEqual((data["tube_minutes"], data["changes"]), (55, 0))
        self.assertIn('Get off at "Heathrow Terminal 5", the stop after "Heathrow Terminals 2 & 3".', data["steps"])
        leg = fixture("pcc_hr5_A")["journeys"][0]["legs"][0]
        self.assertEqual(len(leg["path"]["stopPoints"]), 19)
        # Heathrow loop: the T4-loop train is not one to take for T5.
        self.assertIn('Do not board trains showing "Uxbridge" or "Heathrow Terminal 4".', data["steps"])
        self.assertTrue(any(r.startswith('"Heathrow T123 + 5"') and r.endswith("take this one") for r in data["next_trains"]))

    def test_leicester_square_to_covent_garden_walks(self):
        data, story = run("lsq_cgn", saved_tfl("lsq_cgn"))
        self.assertEqual(data["recommendation"], "walk")
        self.assertEqual(data["walk_minutes"], 6)
        self.assertEqual(data["steps"], ['Walk from "Leicester Square" to "Covent Garden". It takes about 6 minutes.'])
        self.assertEqual((data["changes"], data["next_trains"]), (0, []))

    def test_south_kensington_to_camden_town(self):
        data, _ = run("sks_ctn", saved_tfl("sks_ctn"))
        self.assertEqual((data["tube_minutes"], data["changes"]), (25, 1))
        steps = data["steps"]
        self.assertIn('At "South Kensington", follow signs for the "Piccadilly" line, platform sign "Eastbound".', steps)
        self.assertIn('Change at "Leicester Square".', steps)
        self.assertIn('At "Leicester Square", follow signs for the "Northern" line.', steps)
        self.assertEqual(steps[-1], 'Get off at "Camden Town", 6 stops later.')
        # The live board ticked a short working: it is in the list to board.
        self.assertIn('Board a train showing "Cockfosters" or "Arnos Grove".', steps)

    def test_live_choice_matches_the_javascript(self):
        for key, want in TRUTH["journeys"].items():
            with self.subTest(key):
                parsed = []
                for x in "ABC":
                    body = fixture(f"{key}_{x}")
                    parsed.append(live.parse_journeys(body) if "journeys" in body else [])
                c = live.choose_live(*parsed)
                self.assertEqual("train" if c.kind == "tube" else "walk", want["kind"])
                if c.kind == "walk":
                    self.assertEqual(c.walk_minutes, want["minutes"])
                    continue
                self.assertEqual(c.primary.minutes, want["minutes"])
                self.assertEqual([{"lines": l.lines, "board": l.board_id, "alight": l.alight_id}
                                  for l in c.primary.tube_legs], want["legs"])
                for leg, leave in zip(c.primary.tube_legs, want["leaves"]):
                    if leave["stops"] is not None:
                        self.assertEqual(guide.leave_text(leg), f'Get off at "{leave["station"]}", {leave["stops"]} stops later.')
                    else:
                        self.assertEqual(guide.leave_text(leg), f'Get off at "{leave["station"]}", the stop after "{leave["after"]}".')


# ── The same four journeys with no network at all ───────────────────────────

class StaticJourneys(NoNetwork):
    def lines_and_changes(self, key):
        data, story = run(key, None)
        self.assertFalse(data["uses_live_data"])
        self.assertEqual(data["next_trains"], [])
        self.assertIn("offline estimate", story) if data["recommendation"] == "tube" else None
        return data

    def test_same_lines_and_change_points_as_live(self):
        for key, want in TRUTH["journeys"].items():
            with self.subTest(key):
                a, b = JOURNEYS[key]
                choice = guide.plan_static(guide.resolve_place(a), guide.resolve_place(b))
                self.assertEqual("train" if choice.kind == "tube" else "walk", want["kind"])
                if choice.kind == "tube":
                    got = [{"lines": l.lines, "board": l.board_id, "alight": l.alight_id} for l in choice.primary.tube_legs]
                    self.assertEqual(got, want["legs"])
                    self.assertEqual([len(l.stop_ids) for l in choice.primary.tube_legs],
                                     [x["stops"] or 19 for x in want["leaves"]])

    def test_static_steps_read_like_the_live_ones(self):
        data = self.lines_and_changes("vic_nhg")
        self.assertEqual(data["steps"], [
            'At "Victoria", follow signs for the "Circle" line.',
            'Board a train showing "Edgware Road".',
            'Do not board trains showing "Wimbledon", "Ealing Broadway" or "Richmond".',
            'Get off at "Notting Hill Gate", 5 stops later.',
        ])
        self.assertEqual(data["service_status"], "Not checked: live TfL status was unavailable.")
        self.assertEqual(data["data_note"], "Powered by TfL Open Data.")

    def test_heathrow_uses_the_stop_after_wording_offline(self):
        data = self.lines_and_changes("pcc_hr5")
        self.assertIn('Get off at "Heathrow Terminal 5", the stop after "Heathrow Terminals 2 & 3".', data["steps"])

    def test_minutes_are_estimated_from_the_stated_rule(self):
        # 5 stops x 2 + 3 waiting + 4 platform access = 17
        self.assertEqual(self.lines_and_changes("vic_nhg")["tube_minutes"], 17)
        # 2 legs (5 + 6 stops) x 2 + 7 + one change of 5 = 34 (the 8-minute preference is not reported)
        data = self.lines_and_changes("sks_ctn")
        self.assertEqual((data["tube_minutes"], data["changes"]), (34, 1))

    def test_one_node_per_branch_no_free_branch_switch(self):
        # Charing Cross branch all the way: Mornington Crescent is on the way.
        j = guide.route_static(["940GZZLULSQ"], ["940GZZLUCTN"], guide.CHANGE_PENALTY)
        self.assertEqual(j.tube_legs[0].stop_names[-2:], ["Mornington Crescent", "Camden Town"])

    def test_walk_rule_offline(self):
        data = self.lines_and_changes("lsq_cgn")
        self.assertEqual(data["recommendation"], "walk")


# ── Which trains to board (board.js) ────────────────────────────────────────

class Boards(NoNetwork):
    FILES = {"vic_nhg": "vic_nhg_arrivals", "pcc_hr5": "pcc_hr5_arrivals", "sks_ctn": "sks_ctn_arrivals",
             "ctn_wlo": "ctn_arrivals", "ctn_bnk": "ctn_arrivals"}

    def test_every_mark_and_board_matches_the_javascript(self):
        for key, want in TRUTH["boards"].items():
            with self.subTest(key):
                arrivals = live.parse_arrivals(fixture(self.FILES[key]))
                marks = [{"id": p["id"], "mark": guide.classify(p, want["leg"])}
                         for p in arrivals if p["naptanId"] == want["leg"]["boardId"]]
                self.assertEqual(marks, want["marks"])
                self.assertEqual(guide.build_board(arrivals, want["leg"], names=NAMES), want["board"])

    def test_shared_platform_circle_ticked_district_crossed(self):
        board = TRUTH["boards"]["vic_nhg"]["board"]
        self.assertEqual([(r["lineId"], r["mark"]) for r in board["rows"]],
                         [("district", "skip"), ("circle", "take"), ("district", "skip")])

    def test_northern_wrong_branch_is_listed_to_avoid(self):
        leg = guide.route_static(["940GZZLUCTN"], ["940GZZLUWLO"], guide.CHANGE_PENALTY).tube_legs[0]
        show, avoid = guide.boarding_signs(leg)
        self.assertIn("Morden via Charing Cross", show)
        self.assertEqual(avoid, ["Morden via Bank"])
        leg = guide.route_static(["940GZZLUCTN"], ["940GZZLUBNK"], guide.CHANGE_PENALTY).tube_legs[0]
        show, avoid = guide.boarding_signs(leg)
        self.assertIn("Morden via Bank", show)
        self.assertIn("Morden via Charing Cross", avoid)

    def test_northern_live_marks_follow_the_via(self):
        arrivals = [p for p in live.parse_arrivals(fixture("ctn_arrivals")) if p.get("direction") == "inbound"]
        leg = {"lines": ["northern"], "boardId": "940GZZLUCTN", "alightId": "940GZZLUWLO"}
        marks = {p["towards"]: guide.classify(p, leg) for p in arrivals}
        self.assertEqual(marks["Morden via Bank"], "skip")
        self.assertEqual(marks["Kennington via CX"], "take")

    def test_helpers(self):
        self.assertEqual([guide.minutes_of(s) for s in (0, 29, 30, 89, 90)], [1, 1, 1, 1, 2])
        self.assertEqual(guide.split_platform("Westbound - Platform 1"), {"direction": "Westbound", "platform": "Platform 1"})
        self.assertEqual(guide.split_platform("B"), {"direction": None, "platform": "Platform B"})
        one = {"id": "1", "lineId": "x", "platformName": "A", "destinationNaptanId": "D", "timeToStation": 60}
        self.assertEqual(len(guide.dedupe([one, {**one, "id": "2", "timeToStation": 100}])), 1)
        self.assertEqual(len(guide.dedupe([one, {**one, "id": "2", "timeToStation": 400}])), 2)


# ── Place matching (names.js, find.js) ──────────────────────────────────────

class Matching(unittest.TestCase):
    def test_station_with_or_without_station(self):
        self.assertEqual(guide.resolve_place("victoria station").stations, ["940GZZLUVIC"])
        self.assertEqual(guide.resolve_place("Victoria").stations, ["940GZZLUVIC"])

    def test_typo(self):
        self.assertEqual(guide.resolve_place("lester square").name, "Leicester Square")

    def test_landmark_in_english_and_japanese(self):
        for text in ("british museum", "大英博物館", "大英博物馆", "Британский музей"):
            with self.subTest(text):
                p = guide.resolve_place(text)
                self.assertEqual((p.landmark, p.name), ("british-museum", "British Museum"))
                self.assertTrue(p.stations)

    def test_station_name_beats_landmark_of_the_same_name(self):
        self.assertIsNone(guide.resolve_place("Piccadilly Circus").landmark)
        self.assertIsNone(guide.resolve_place("Covent Garden").landmark)

    def test_landmark_uses_the_best_nearby_station(self):
        bm = guide.resolve_place("British Museum")
        self.assertGreater(len(bm.options), 1)
        o, d = guide.best_access(guide.resolve_place("Victoria"), bm)
        totals = []
        for st, walk in bm.options:
            j = guide.route_static(["940GZZLUVIC"], st, guide.CHANGE_PENALTY)
            totals.append(j.minutes + walk + guide.CHANGE_PENALTY * j.changes)
        chosen = bm.options[[st for st, _ in bm.options].index(d.stations)]
        j = guide.route_static(["940GZZLUVIC"], d.stations, guide.CHANGE_PENALTY)
        self.assertEqual(j.minutes + chosen[1] + guide.CHANGE_PENALTY * j.changes, min(totals))
        self.assertEqual(o.stations, ["940GZZLUVIC"], "a station stays itself")

    def test_far_landmark_is_honest(self):
        data, story = live.answer(guide.resolve_place("Bank"), guide.resolve_place("London City Airport"), None)
        self.assertTrue(data["steps"][-1].startswith('The Tube does not go close to "London City Airport".'))
        self.assertIn('"DLR"', data["steps"][-1])
        self.assertIn('"DLR"', story)

    def test_twin_stations_are_one_place(self):
        self.assertEqual(len(guide.resolve_place("Edgware Road").stations), 2)
        self.assertEqual(len(guide.resolve_place("Hammersmith").stations), 2)
        self.assertEqual(len(guide.resolve_place("Paddington").stations), 2)

    def test_errors(self):
        with self.assertRaises(guide.UnknownPlace):
            guide.resolve_place("Narnia Central")
        with self.assertRaises(guide.MissingPlace):
            guide.resolve_place("  ")

    def test_names(self):
        self.assertEqual(guide.fold_name("King's Cross St. Pancras"), "kings cross st pancras")
        self.assertEqual(guide.clean_station_name("Battersea Power Station Underground Station"), "Battersea Power Station")
        self.assertEqual(guide.station_sign("940GZZLUERC"), "Edgware Road")
        self.assertEqual(guide.station_sign("940GZZLUKOY"), "Kensington (Olympia)")


# ── Live failures fall back to the static answer ────────────────────────────

class LiveFailures(NoNetwork):
    def assert_static(self, get_json):
        data, _ = run("vic_nhg", get_json)
        self.assertFalse(data["uses_live_data"])
        self.assertEqual(data["next_trains"], [])
        self.assertEqual(data["tube_minutes"], 17)            # the static estimate
        self.assertIn('Get off at "Notting Hill Gate", 5 stops later.', data["steps"])

    def test_raise(self):
        def boom(url, timeout):
            raise RuntimeError("anything at all")
        self.assert_static(boom)

    def test_timeout(self):
        def slow(url, timeout):
            raise TimeoutError("timed out")
        self.assert_static(slow)

    def test_403_cloudflare(self):
        def blocked(url, timeout):
            raise urllib.error.HTTPError(url, 403, "error code: 1010", {}, None)
        self.assert_static(blocked)

    def test_unreachable(self):
        def offline(url, timeout):
            raise urllib.error.URLError("no route")
        self.assert_static(offline)

    def test_empty_and_odd_bodies(self):
        for body in ({}, [], {"journeys": []}, {"journeys": [{"legs": [{"mode": {"id": "boat"}}]}]}, "nonsense", None):
            with self.subTest(body=body):
                self.assert_static(lambda url, timeout, body=body: body)

    def test_status_and_arrivals_may_fail_alone(self):
        good = saved_tfl("vic_nhg")

        def planner_only(url, timeout):
            if "/Journey/" in url:
                return good(url, timeout)
            raise TimeoutError()
        data, _ = run("vic_nhg", planner_only)
        self.assertTrue(data["uses_live_data"])
        self.assertEqual(data["tube_minutes"], 13)
        self.assertEqual(data["next_trains"], [])
        self.assertEqual(data["service_status"], "Not checked: live TfL status was unavailable.")


class Status(unittest.TestCase):
    def test_three_states(self):
        s = fixture("status_all")
        self.assertEqual(guide.status_strip(s, ["circle"])["state"], "ok")
        self.assertEqual(guide.status_sentence(guide.status_strip(s, ["circle"])), '"Good Service" on every line in this journey.')
        self.assertEqual(guide.status_sentence(guide.status_strip(s, ["district"])), '"District": "Minor Delays". Allow extra time.')
        closed = [{"id": "x", "name": "X", "lineStatuses": [{"statusSeverityDescription": "Part Suspended"}]}]
        self.assertEqual(guide.status_strip(closed, ["x"])["state"], "closed")


class Contract(unittest.TestCase):
    """What can be checked about the Pippa contract without the SDK."""

    def test_outputs_match_the_manifest(self):
        import re
        text = (HERE / "manifest.yaml").read_text("utf-8")
        declared = re.findall(r"^      - name: (\w+)", text.split("output:")[1], re.M)
        data, _ = run("vic_nhg", None)
        self.assertEqual(sorted(declared), sorted(data))
        types = {"recommendation": str, "steps": list, "tube_minutes": int, "walk_minutes": int, "changes": int,
                 "next_trains": list, "service_status": str, "uses_live_data": bool, "data_note": str}
        for key, kind in types.items():
            self.assertIsInstance(data[key], kind, key)
            if kind is int:
                self.assertNotIsInstance(data[key], bool, key)
        self.assertTrue(all(isinstance(s, str) for s in data["steps"] + data["next_trains"]))

    def test_summary_never_says_by_and_no_key_anywhere(self):
        import re
        text = (HERE / "manifest.yaml").read_text("utf-8")
        summary = text.split("summary: >-")[1].split("\n\n")[0]
        self.assertNotRegex(summary.lower(), r"\bby\b")
        for path in HERE.iterdir():
            if path.suffix in (".py", ".yaml", ".json"):
                body = path.read_text("utf-8")
                self.assertNotIn("app_key=", body.replace('"app_key="', ""), path.name)
                self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", body), f"email-like text in {path.name}")

    def test_flat_folder(self):
        # The starter repo's own folders are not deployed; the app's files must not live in a subfolder.
        not_deployed = {"__pycache__", "docs", "scripts"}
        self.assertEqual([p.name for p in HERE.iterdir()
                          if p.is_dir() and p.name not in not_deployed and not p.name.startswith(".")], [])


if __name__ == "__main__":
    unittest.main()
