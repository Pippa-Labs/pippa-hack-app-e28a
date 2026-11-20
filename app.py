"""Tube Guide: which train to take, where to change, when to get off.

Thin on purpose. The logic is in guide.py (pure, no network) and live.py
(pure parsing of TfL answers), so almost all of it is tested without the SDK.

It always answers without the network: the static router in guide.py works
from routes.json. Live TfL data is an extra on top. Measured on 2 Nov 2026,
api.tfl.gov.uk answered Pippa's servers with 403 "error code: 1010"
(docs/IDEAS.md), so any live failure (exception, timeout, non-200, an empty
or odd body) falls back silently to the static answer.

No TfL key anywhere: TfL is called anonymously.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    Result,
    load_manifest_file,
    validate_manifest,
)

import guide
import live

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")

#: TfL's edge refuses some anonymous clients; urllib's own User-Agent is the
#: likeliest reason, so every request says what it is.
USER_AGENT = "pippa-agent-app-tube-guide/0.1 (https://github.com/Pippa-Labs)"



class MissingPlaces(AgentAppError):
    user_message = ("Tell me where you are starting and where you are going, "
                    "for example Victoria to Notting Hill Gate.")
    retryable = False


class UnknownPlace(AgentAppError):
    user_message = "I could not find a Tube station or landmark with that name."
    retryable = False


class SamePlace(AgentAppError):
    user_message = "Those are the same station."
    retryable = False


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # ── The one network seam (tests replace it on the instance) ─────────────
    def _fetch(self, url: str, timeout: float = 6.0) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _json(self, url: str, timeout: float):
        return json.loads(self._fetch(url, timeout=timeout).decode("utf-8"))

    # ── handle ───────────────────────────────────────────────────────────────
    def handle(self, call: Call) -> Result:
        if call.outcome_id != "tube.route":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")
        origin = self._place(call.inputs.get("from_place"))
        dest = self._place(call.inputs.get("to_place"))
        if origin.key == dest.key:
            raise SamePlace(f"same place: {origin.key}")
        # Static answer first, live TfL on top; live.answer never raises for a
        # live failure (it falls back to the static answer).
        data, narrative = live.answer(origin, dest, self._json)
        return Result(data=data, narrative=narrative)

    @staticmethod
    def _place(text) -> guide.Place:
        text = (text or "").strip()
        if not text:
            raise MissingPlaces("from_place or to_place is empty")
        try:
            return guide.resolve_place(text)
        except guide.UnknownPlace:
            error = UnknownPlace(f"no match for {text!r}")
            error.user_message = f"I could not find a Tube station or landmark called {text}."
            raise error
