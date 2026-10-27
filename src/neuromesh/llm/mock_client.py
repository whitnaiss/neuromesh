"""Placeholder mock client so every track can run without a GPU.

Track B (Armman) owns this file and will replace it with the deterministic,
rule-based mock (forward everything known, take the local majority).
"""

from __future__ import annotations

import json

from neuromesh.contracts import LLMRequest, LLMResponse


class MockLLMClient:
    """Returns a valid, empty AgentTurn for every request."""

    def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        empty_turn = json.dumps({"messages": {}, "state": {}})
        return [LLMResponse(agent_id=r.agent_id, text=empty_turn) for r in requests]
