"""Codex backend — `codex exec` on ChatGPT subscription auth.

Differences from the Claude Code backend that shape this file:

- Codex has a real **`--output-schema <FILE>`** flag, so when a call forces a
  single `record_*` tool we constrain the final message with that schema and
  read the answer from `--output-last-message`. It is stricter and cheaper
  than a tool round-trip. The MCP capture path remains as the fallback, and
  research tools always go through MCP so provenance still works.
- Output is **JSONL events**, not one object: `thread.started`,
  `turn.started`, `item.completed`, `turn.completed`, `turn.failed`, `error`.
- MCP servers are configured through repeated `-c mcp_servers.<name>.<key>`
  overrides whose values are parsed as TOML.
- Sandboxing defaults to `read-only`: this backend only ever needs to think
  and call our tools, never to edit the working tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import Config
from ..retry import classify_failure
from ..types import AgentCallSpec, CallContext
from .base import AgentCliProvider, CliInvocation, CliOutcome

LAST_MESSAGE_FILE = "last-message.txt"
FORCED_RECORD_FILE = "_forced_record.txt"

# Reasoning effort tiers, mirroring the Claude backend's mapping.
_EFFORT_TIERS: tuple[tuple[int, str], ...] = (
    (1, "low"),
    (4_000, "medium"),
    (8_000, "high"),
    (16_000, "xhigh"),
)


def effort_for_thinking_tokens(thinking_tokens: int) -> str:
    for ceiling, tier in _EFFORT_TIERS:
        if thinking_tokens <= ceiling:
            return tier
    return "xhigh"


class CodexCliProvider(AgentCliProvider):
    backend_name = "codex_cli"

    def _resolve_backend_cfg(self, cfg: Config) -> Any:
        return cfg.llm.codex_cli

    # ----------------------------- command ----------------------------- #

    def build_invocation(
        self, spec: AgentCallSpec, ctx: CallContext, capture_dir: Path
    ) -> CliInvocation:
        record_tools = [t["name"] for t in spec.tools if _is_record_tool(t)]
        research_tools = [t["name"] for t in spec.tools if not _is_record_tool(t)]
        mcp_agent = ctx.agent if research_tools else ""

        argv = [
            self._binary, "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s", self._backend_cfg.sandbox,
            "-m", spec.route.model,
            "-o", str(capture_dir / LAST_MESSAGE_FILE),
            "-c", _toml_kv(
                "model_reasoning_effort",
                effort_for_thinking_tokens(spec.route.thinking_tokens),
            ),
        ]

        # Structured output: native schema when exactly one record is expected.
        forced = _forced_record_tool(spec, record_tools)
        if forced is not None:
            schema = next(
                t["input_schema"] for t in spec.tools if t["name"] == forced
            )
            schema_path = capture_dir / "output-schema.json"
            schema_path.write_text(
                json.dumps(_strict_object_schema(schema)), encoding="utf-8"
            )
            (capture_dir / FORCED_RECORD_FILE).write_text(forced, encoding="utf-8")
            argv += ["--output-schema", str(schema_path)]

        if record_tools or research_tools:
            mcp = self.mcp_config(
                capture_dir=capture_dir,
                record_tools=record_tools,
                agent=mcp_agent,
                ctx=ctx,
            )
            argv += _mcp_overrides(self.server_label, mcp["mcpServers"][self.server_label])

        return CliInvocation(argv=argv, stdin=_build_prompt(spec, forced))

    # ----------------------------- parsing ----------------------------- #

    def parse_outcome(
        self, *, stdout: str, stderr: str, returncode: int, capture_dir: Path
    ) -> CliOutcome:
        events = _parse_events(stdout)
        failure = _find_failure(events)
        if failure:
            raise classify_failure(f"codex error: {failure[:800]}")
        if not events and returncode != 0:
            detail = (stderr or stdout or "no output").strip()[:800]
            raise classify_failure(f"codex exited {returncode}: {detail}")

        text = _read_last_message(capture_dir) or _assistant_text(events)

        # Native schema output arrives as the final message rather than a tool
        # call. Normalize it into the capture dir so downstream code sees one
        # uniform shape.
        forced_path = capture_dir / FORCED_RECORD_FILE
        if forced_path.exists() and not list(capture_dir.glob("record-*.json")):
            payload = _parse_json_object(text)
            if payload is not None:
                tool = forced_path.read_text(encoding="utf-8").strip()
                (capture_dir / f"record-0001-{tool}.json").write_text(
                    json.dumps({"tool": tool, "seq": 1, "input": payload}),
                    encoding="utf-8",
                )

        usage = _find_usage(events)
        return CliOutcome(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read=usage.get("cached_input_tokens", 0),
            num_turns=max(1, sum(1 for e in events if e.get("type") == "turn.started")),
            reported_cost_usd=None,        # Codex reports tokens, not dollars
            raw={"events": events[-50:]},  # tail is enough to debug a failure
        )


# --------------------------------------------------------------------------- #
# command helpers


def _mcp_overrides(label: str, server: dict[str, Any]) -> list[str]:
    """Render an MCP server definition as `-c mcp_servers.<label>.*` flags."""
    prefix = f"mcp_servers.{label}"
    argv = [
        "-c", _toml_kv(f"{prefix}.command", server["command"]),
        "-c", f'{prefix}.args={_toml_value(server["args"])}',
    ]
    if server.get("cwd"):
        argv += ["-c", _toml_kv(f"{prefix}.cwd", server["cwd"])]
    if server.get("env"):
        argv += ["-c", f'{prefix}.env={_toml_value(server["env"])}']
    return argv


def _toml_kv(key: str, value: Any) -> str:
    return f"{key}={_toml_value(value)}"


def _toml_value(value: Any) -> str:
    """Render a Python value as a TOML literal.

    JSON is a close enough subset for the scalars, arrays, and single-level
    string tables we pass (TOML inline tables use the same `{k = v}` shape
    once keys are quoted).
    """
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in value.items()
        ) + "}"
    return json.dumps(str(value))


def _forced_record_tool(spec: AgentCallSpec, record_tools: list[str]) -> str | None:
    """The single record tool this call must produce, if there is exactly one."""
    if spec.tool_choice and spec.tool_choice.get("type") == "tool":
        name = spec.tool_choice.get("name")
        return name if name in record_tools else None
    return record_tools[0] if len(record_tools) == 1 else None


def _strict_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Codex expects a self-contained JSON Schema for the final response."""
    out = _strict_schema(schema)
    out.setdefault("type", "object")
    out.setdefault("additionalProperties", False)
    return out


def _strict_schema(schema: Any) -> Any:
    """Adapt tool schemas to Codex strict-output rules."""
    if not isinstance(schema, dict):
        return schema

    out = dict(schema)
    properties = out.get("properties")
    if isinstance(properties, dict):
        out["type"] = "object"
        out["additionalProperties"] = False
        out["properties"] = {
            name: _strict_schema(value) for name, value in properties.items()
        }
        # Codex requires every declared property to appear in `required`.
        out["required"] = list(properties)

    if "items" in out:
        out["items"] = _strict_schema(out["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = out.get(keyword)
        if isinstance(variants, list):
            out[keyword] = [_strict_schema(value) for value in variants]
    return out


def _build_prompt(spec: AgentCallSpec, forced: str | None) -> str:
    from .claude_code import build_prompt, build_system_prompt

    # Codex has no separate system-prompt flag on `exec`; prepend it.
    system = build_system_prompt(spec)
    body = build_prompt(spec)
    parts = [p for p in (system, body) if p]
    if forced is not None:
        parts.append(
            "## Required output\n"
            "Respond with a single JSON object matching the provided output "
            "schema. No prose, no code fences."
        )
    return "\n\n".join(parts)


def _is_record_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("name", "")).startswith("record_")


# --------------------------------------------------------------------------- #
# event-stream parsing


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _find_failure(events: list[dict[str, Any]]) -> str | None:
    """Return the fatal error message, if the turn failed.

    `item.completed` items of type "error" are advisory warnings (unsupported
    service tier, missing model metadata) and must not fail the call — only
    top-level `error` and `turn.failed` are terminal.
    """
    for event in events:
        etype = event.get("type")
        if etype == "turn.failed":
            err = event.get("error") or {}
            return str(err.get("message") or "turn failed")
        if etype == "error":
            return str(event.get("message") or "unknown error")
    return None


def _find_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Pull token counts from whichever event carries them."""
    out: dict[str, int] = {}
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            info = event.get("info")
            usage = info.get("usage") if isinstance(info, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                out[key] = value
    return out


def _assistant_text(events: list[dict[str, Any]]) -> str:
    """Fallback when `--output-last-message` produced nothing."""
    chunks: list[str] = []
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") in ("assistant_message", "agent_message", "message"):
            text = item.get("text") or item.get("message") or ""
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n\n".join(chunks).strip()


def _read_last_message(capture_dir: Path) -> str:
    path = capture_dir / LAST_MESSAGE_FILE
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a ```json fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1]
        if body.endswith("```"):
            body = body[: -3]
        body = body.strip()
    start = body.find("{")
    if start == -1:
        return None
    try:
        parsed = json.loads(body[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["CodexCliProvider", "effort_for_thinking_tokens"]
