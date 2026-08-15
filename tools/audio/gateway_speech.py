"""Text-to-speech through the Vercel AI Gateway's native speech protocol.

Premium narration voices (OpenAI tts-1 family) behind the same single
gateway key the rest of generation uses. The OpenAI-dialect
/v1/audio/speech endpoint does not exist on the gateway; this speaks the
native surface, proven live:

    POST {AI_GATEWAY_SPEECH_URL}
    authorization: Bearer $OPENAI_API_KEY
    ai-gateway-protocol-version: 0.0.1
    ai-gateway-auth-method: api-key
    ai-model-id: openai/tts-1
    {"text": ..., "voice": ..., "outputFormat": "mp3"}

Response: JSON {"audio": "<base64 mp3>"}.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_PER_MILLION_CHARS_USD = 15.0  # openai tts-1 published rate


def _endpoint() -> str | None:
    explicit = os.environ.get("AI_GATEWAY_SPEECH_URL")
    if explicit:
        return explicit
    base = os.environ.get("OPENAI_BASE_URL", "")
    if "ai-gateway.vercel.sh" in base:
        return base.replace("/v1", "").rstrip("/") + "/v4/ai/speech-model"
    return None


def _auth_headers() -> dict[str, str]:
    """Authorization for gateway egress. Under GATEWAY_AUTH=firewall the
    sandbox's egress firewall injects the credential for the gateway
    domain — the key never exists inside the sandbox — so no header is
    sent from here."""
    if os.environ.get("GATEWAY_AUTH") == "firewall":
        return {}
    key = os.environ.get("OPENAI_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _authed() -> bool:
    return os.environ.get("GATEWAY_AUTH") == "firewall" or bool(os.environ.get("OPENAI_API_KEY"))


class GatewaySpeech(BaseTool):
    name = "gateway_speech"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "text_to_speech"
    provider = "ai-gateway"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set OPENAI_API_KEY to a Vercel AI Gateway key and point\n"
        "OPENAI_BASE_URL at https://ai-gateway.vercel.sh/v1 (or set\n"
        "AI_GATEWAY_SPEECH_URL directly)."
    )
    agent_skills = []

    capabilities = ["narration", "premium_voices", "gateway_routed"]
    supports = {"voice_choice": True, "mp3_output": True}
    best_for = [
        "premium narration voices (alloy, echo, fable, onyx, nova, shimmer) behind the one gateway key",
        "warmer reads than local Piper when the plan calls for it",
    ]
    not_good_for = ["offline use", "singing or music"]
    fallback_tools = ["piper_tts"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "The narration to speak."},
            "voice": {
                "type": "string",
                "enum": ["alloy", "ash", "echo", "fable", "onyx", "nova", "shimmer"],
                "default": "onyx",
            },
            "model": {
                "type": "string",
                "enum": ["openai/tts-1", "openai/tts-1-hd"],
                "default": "openai/tts-1",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "model"]
    side_effects = ["writes mp3 to output_path", "calls the AI Gateway"]
    user_visible_verification = ["Listen to the narration for tone and clarity"]

    def get_status(self) -> ToolStatus:
        if _authed() and _endpoint():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        chars = len(str(inputs.get("text", "")))
        rate = _PER_MILLION_CHARS_USD * (2 if "hd" in str(inputs.get("model", "")) else 1)
        return round(chars / 1_000_000 * rate, 6)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import requests  # lazy: the registry must load without third-party deps

        start = time.time()
        endpoint = _endpoint()
        if not endpoint or not _authed():
            return ToolResult(
                success=False,
                error="gateway_speech needs gateway auth (OPENAI_API_KEY or GATEWAY_AUTH=firewall) and a gateway endpoint.",
            )

        text = str(inputs.get("text", "")).strip()
        if not text:
            return ToolResult(success=False, error="text is required")

        try:
            response = requests.post(
                endpoint,
                headers={
                    **_auth_headers(),
                    "ai-gateway-protocol-version": "0.0.1",
                    "ai-gateway-auth-method": "api-key",
                    "ai-model-id": str(inputs.get("model", "openai/tts-1")),
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "voice": inputs.get("voice", "onyx"),
                    "outputFormat": "mp3",
                },
                timeout=120,
            )
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"Gateway speech request failed: {exc}")
        if response.status_code != 200:
            return ToolResult(success=False, error=f"Gateway speech HTTP {response.status_code}: {response.text[:300]}")

        audio_b64 = (response.json() or {}).get("audio", "")
        if not audio_b64:
            return ToolResult(success=False, error="Gateway speech returned no audio.")

        output_path = Path(inputs.get("output_path") or f"gateway_speech_{abs(hash(text)) % 10**8}.mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(audio_b64))

        return ToolResult(
            success=True,
            data={
                "provider": "ai-gateway",
                "model": inputs.get("model", "openai/tts-1"),
                "voice": inputs.get("voice", "onyx"),
                "output": str(output_path),
                "characters": len(text),
                "format": "mp3",
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
        )
