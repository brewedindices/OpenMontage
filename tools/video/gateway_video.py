"""Video generation through the Vercel AI Gateway's native video protocol.

One tool, many models: the gateway serves dozens of video models (Veo,
Kling, Seedance, Wan, Sora, ...) behind a single API key. The wire
protocol is the AI SDK's gateway video call, spoken directly:

    POST {AI_GATEWAY_VIDEO_URL}
    authorization: Bearer $OPENAI_API_KEY
    ai-gateway-protocol-version: 0.0.1
    ai-gateway-auth-method: api-key
    ai-video-model-specification-version: 4
    ai-model-id: <model>
    accept: text/event-stream

    {"prompt": ..., "duration": ..., "resolution": ..., "aspectRatio": ...}

The response is an SSE stream whose first data event carries the finished
video as a URL or base64 payload.

Environment:
    OPENAI_API_KEY            gateway key (reused; one key, one gateway)
    AI_GATEWAY_VIDEO_URL      full endpoint URL; when unset it is derived
                              from OPENAI_BASE_URL when that points at
                              ai-gateway.vercel.sh
    GATEWAY_VIDEO_MODELS      comma-separated allowed model ids
    GATEWAY_VIDEO_PRICING     JSON {model: {"per_second": usd}} — real
                              prices from the gateway's /v1/models
"""

from __future__ import annotations

import base64
import json
import os
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

_FALLBACK_PER_SECOND = 0.05  # conservative placeholder when pricing env lacks the model


def _endpoint() -> str | None:
    explicit = os.environ.get("AI_GATEWAY_VIDEO_URL")
    if explicit:
        return explicit
    base = os.environ.get("OPENAI_BASE_URL", "")
    if "ai-gateway.vercel.sh" in base:
        return base.replace("/v1", "").rstrip("/") + "/v4/ai/video-model"
    return None


def _models() -> list[str]:
    raw = os.environ.get("GATEWAY_VIDEO_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _pricing() -> dict[str, Any]:
    try:
        return json.loads(os.environ.get("GATEWAY_VIDEO_PRICING", "{}"))
    except json.JSONDecodeError:
        return {}


class GatewayVideo(BaseTool):
    name = "gateway_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "ai-gateway"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set OPENAI_API_KEY to a Vercel AI Gateway key and point\n"
        "OPENAI_BASE_URL at https://ai-gateway.vercel.sh/v1 (or set\n"
        "AI_GATEWAY_VIDEO_URL directly). List allowed models in\n"
        "GATEWAY_VIDEO_MODELS and their prices in GATEWAY_VIDEO_PRICING."
    )
    agent_skills = []

    capabilities = ["text_to_video", "multi_model", "gateway_routed"]
    supports = {
        "model_choice": True,
        "aspect_ratio": True,
        "resolution": True,
        "audio": True,
        "seed": True,
    }
    best_for = [
        "one key, many video models — Veo, Kling, Seedance, Wan and more",
        "cost-ranged provider choice with real per-second prices",
        "environments where only the gateway key exists",
    ]
    not_good_for = [
        "image-to-video with local reference files (not wired yet)",
        "offline use",
    ]
    fallback_tools = ["pixabay_video", "pexels_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "model"],
        "properties": {
            "prompt": {"type": "string", "description": "What the clip shows."},
            "model": {
                "type": "string",
                "description": "A gateway video model id from GATEWAY_VIDEO_MODELS, e.g. google/veo-3.1-fast-generate-001",
            },
            "duration": {"type": "integer", "default": 4, "minimum": 1, "maximum": 15},
            "resolution": {
                "type": "string",
                "enum": ["480p", "720p", "1080p"],
                "default": "720p",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "default": "16:9",
            },
            "generate_audio": {"type": "boolean", "default": False},
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "duration", "resolution", "seed"]
    side_effects = ["writes video file to output_path", "calls the AI Gateway"]
    user_visible_verification = ["Watch the clip to verify it matches the intended beat"]

    def get_status(self) -> ToolStatus:
        if os.environ.get("OPENAI_API_KEY") and _endpoint() and _models():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        seconds = int(inputs.get("duration", 4))
        entry = _pricing().get(str(inputs.get("model", "")), {})
        per_second = entry.get("per_second", _FALLBACK_PER_SECOND)
        try:
            return float(per_second) * seconds
        except (TypeError, ValueError):
            return _FALLBACK_PER_SECOND * seconds

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import requests  # lazy: the registry must load without third-party deps

        endpoint = _endpoint()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not endpoint or not api_key:
            return ToolResult(
                success=False,
                error="gateway_video needs OPENAI_API_KEY and a gateway endpoint (AI_GATEWAY_VIDEO_URL or a gateway OPENAI_BASE_URL).",
            )

        model = str(inputs.get("model", "")).strip()
        allowed = _models()
        if model not in allowed:
            return ToolResult(
                success=False,
                error=f"Model {model!r} is not in GATEWAY_VIDEO_MODELS. Allowed: {', '.join(allowed) or '(none configured)'}",
            )

        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        duration = int(inputs.get("duration", 4))
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n": 1,
            "duration": duration,
            "resolution": inputs.get("resolution", "720p"),
            "aspectRatio": inputs.get("aspect_ratio", "16:9"),
        }
        if inputs.get("generate_audio"):
            payload["generateAudio"] = True
        if inputs.get("seed") is not None:
            payload["seed"] = int(inputs["seed"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-gateway-auth-method": "api-key",
            "ai-video-model-specification-version": "4",
            "ai-model-id": model,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, stream=True, timeout=600
            )
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"Gateway video request failed: {exc}")

        if response.status_code != 200:
            body = response.text[:400]
            return ToolResult(
                success=False,
                error=f"Gateway video returned HTTP {response.status_code}: {body}",
            )

        event: dict[str, Any] | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line and raw_line.startswith("data:"):
                try:
                    event = json.loads(raw_line[5:].strip())
                except json.JSONDecodeError:
                    continue
                break
        if event is None:
            return ToolResult(success=False, error="Gateway video stream ended without a result event.")
        if event.get("type") == "error":
            return ToolResult(
                success=False,
                error=f"Gateway video error: {event.get('message', 'unknown')}",
            )

        videos = event.get("videos") or []
        if not videos:
            return ToolResult(success=False, error="Gateway video returned no videos.")
        video = videos[0]

        output = inputs.get("output_path") or f"gateway_video_{abs(hash((prompt, model))) % 10**8}.mp4"
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if video.get("type") == "url" and video.get("url"):
                download = requests.get(video["url"], timeout=300)
                download.raise_for_status()
                output_path.write_bytes(download.content)
            elif video.get("type") == "base64" and video.get("data"):
                output_path.write_bytes(base64.b64decode(video["data"]))
            else:
                return ToolResult(success=False, error=f"Unrecognized video payload: {list(video.keys())}")
        except (requests.RequestException, ValueError, OSError) as exc:
            return ToolResult(success=False, error=f"Could not save gateway video: {exc}")

        cost = self.estimate_cost(inputs)
        return ToolResult(
            success=True,
            data={
                "provider": "ai-gateway",
                "model": model,
                "prompt": prompt,
                "output": str(output_path),
                "duration": duration,
                "resolution": payload["resolution"],
                "aspect_ratio": payload["aspectRatio"],
                "format": "mp4",
                "warnings": event.get("warnings") or [],
            },
            artifacts=[str(output_path)],
            cost_usd=cost,
        )
