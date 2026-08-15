"""Speech-to-text through the Vercel AI Gateway's native transcription
protocol — real word/segment timings for captions, behind the one key.

The OpenAI-dialect /v1/audio/transcriptions endpoint does not exist on
the gateway; this speaks the native surface, proven live:

    POST {AI_GATEWAY_TRANSCRIPTION_URL}
    authorization: Bearer $OPENAI_API_KEY
    ai-gateway-protocol-version: 0.0.1
    ai-gateway-auth-method: api-key
    ai-model-id: openai/whisper-1
    {"audio": "<base64>", "mediaType": "audio/mp4"}

Response: JSON {"text": ..., "segments": [...], "durationInSeconds": n}.
Caption timing from these segments is speech-aligned, not interpolated.
"""

from __future__ import annotations

import base64
import json
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

_PER_MINUTE_USD = 0.006  # openai whisper-1 published rate

_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def _endpoint() -> str | None:
    explicit = os.environ.get("AI_GATEWAY_TRANSCRIPTION_URL")
    if explicit:
        return explicit
    base = os.environ.get("OPENAI_BASE_URL", "")
    if "ai-gateway.vercel.sh" in base:
        return base.replace("/v1", "").rstrip("/") + "/v4/ai/transcription-model"
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


class GatewayTranscribe(BaseTool):
    name = "gateway_transcribe"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "transcription"
    provider = "ai-gateway"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set OPENAI_API_KEY to a Vercel AI Gateway key and point\n"
        "OPENAI_BASE_URL at https://ai-gateway.vercel.sh/v1 (or set\n"
        "AI_GATEWAY_TRANSCRIPTION_URL directly)."
    )
    agent_skills = []

    capabilities = ["transcription", "segment_timings", "gateway_routed"]
    supports = {"word_timed_captions": True, "audio_and_video_containers": True}
    best_for = [
        "speech-aligned caption timing (replaces interpolated estimates)",
        "verifying narration matches the script",
    ]
    not_good_for = ["offline use", "files over ~24MB of audio"]
    fallback_tools = []

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string", "description": "Audio (or mp4) file to transcribe."},
            "model": {"type": "string", "default": "openai/whisper-1"},
            "output_path": {"type": "string", "description": "Where to write the segments JSON."},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["input_path", "model"]
    side_effects = ["writes segments JSON to output_path", "calls the AI Gateway"]
    user_visible_verification = ["Spot-check segment timings against the audio"]

    def get_status(self) -> ToolStatus:
        if _authed() and _endpoint():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Flat conservative guess; the true rate is $0.006/min of audio.
        return 0.01

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import requests  # lazy: the registry must load without third-party deps

        start = time.time()
        endpoint = _endpoint()
        if not endpoint or not _authed():
            return ToolResult(
                success=False,
                error="gateway_transcribe needs gateway auth (OPENAI_API_KEY or GATEWAY_AUTH=firewall) and a gateway endpoint.",
            )

        source = Path(str(inputs.get("input_path", "")))
        if not source.is_file():
            return ToolResult(success=False, error=f"No file at {source}")
        media_type = _MEDIA_TYPES.get(source.suffix.lower(), "audio/mpeg")

        try:
            response = requests.post(
                endpoint,
                headers={
                    **_auth_headers(),
                    "ai-gateway-protocol-version": "0.0.1",
                    "ai-gateway-auth-method": "api-key",
                    "ai-model-id": str(inputs.get("model", "openai/whisper-1")),
                    "Content-Type": "application/json",
                },
                json={
                    "audio": base64.b64encode(source.read_bytes()).decode(),
                    "mediaType": media_type,
                },
                timeout=300,
            )
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"Gateway transcription request failed: {exc}")
        if response.status_code != 200:
            return ToolResult(success=False, error=f"Gateway transcription HTTP {response.status_code}: {response.text[:300]}")

        body = response.json() or {}
        segments = body.get("segments") or []
        duration = float(body.get("durationInSeconds") or 0.0)

        output_path = None
        if inputs.get("output_path"):
            output_path = Path(str(inputs["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({"text": body.get("text", ""), "segments": segments}, indent=2))

        return ToolResult(
            success=True,
            data={
                "provider": "ai-gateway",
                "model": inputs.get("model", "openai/whisper-1"),
                "text": body.get("text", ""),
                "segment_count": len(segments),
                "segments": segments[:50],
                "duration_seconds": duration,
                **({"output": str(output_path)} if output_path else {}),
            },
            artifacts=[str(output_path)] if output_path else [],
            cost_usd=round(max(duration, 1.0) / 60 * _PER_MINUTE_USD, 6),
            duration_seconds=round(time.time() - start, 2),
        )
