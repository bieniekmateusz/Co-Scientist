"""Tests for the CLI-driven backends.

Everything here runs against a fake `claude` binary, so the suite is
deterministic and costs no subscription quota. The one test that matters most
is `test_api_keys_are_stripped_from_the_subprocess_environment`: if that
regresses, the project silently goes back to metered API billing, which is
precisely what this backend exists to avoid.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from co_scientist.config import Config
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.cli_backend.base import (
    BILLING_ENV_VARS,
    build_message,
    read_capture,
    sanitized_env,
)
from co_scientist.llm.cli_backend.claude_code import (
    ClaudeCliProvider,
    build_prompt,
    effort_for_thinking_tokens,
)
from co_scientist.llm.cli_backend.codex import _strict_object_schema
from co_scientist.llm.retry import CliBackendError, CliRetryableError
from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.types import (
    AgentCallSpec,
    CachedBlock,
    CallContext,
    CaptureBundle,
)

RECORD_TOOL = {
    "name": "record_hypothesis",
    "description": "record it",
    "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
}
SEARCH_TOOL = {"name": "pubmed_search", "description": "search", "input_schema": {}}


# --------------------------------------------------------------------------- #
# fixtures


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    """An executable that impersonates `claude -p`."""
    from co_scientist.tests.fakes import fake_claude as module

    launcher = tmp_path / "fake-claude"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{Path(module.__file__)}" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


@pytest.fixture
def cli_cfg(tmp_cfg: Config, fake_claude: Path) -> Config:
    tmp_cfg.llm.provider = "claude_cli"
    tmp_cfg.llm.claude_cli.binary = str(fake_claude)
    tmp_cfg.llm.claude_cli.timeout_seconds = 30
    return tmp_cfg


@pytest.fixture
async def provider(cli_cfg: Config, conn) -> ClaudeCliProvider:
    from co_scientist.models import ResearchPlan, Session
    from co_scientist.storage.repos import sessions as sess_repo

    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id="ses_test", created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="g"),
        config_snapshot={}, budget_tokens=1_000_000, budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    ))
    budget = TokenBudget(cfg=cli_cfg, budget_tokens=1_000_000, budget_usd=10.0)
    return ClaudeCliProvider(cli_cfg, db=conn, budget=budget)


def _spec(**kw: Any) -> AgentCallSpec:
    return AgentCallSpec(
        route=kw.pop("route", ModelRoute(
            agent="generation", mode="literature", model="opus", thinking_tokens=4000,
        )),
        system_blocks=kw.pop("system_blocks", [CachedBlock("You are the generation agent.")]),
        user_blocks=kw.pop("user_blocks", [CachedBlock("Find a hypothesis.")]),
        tools=kw.pop("tools", [SEARCH_TOOL, RECORD_TOOL]),
        **kw,
    )


def _ctx(task_id: str | None = None) -> CallContext:
    # task_id stays None: transcripts carry a FK to tasks, and these tests
    # exercise the backend rather than the scheduler.
    return CallContext(
        session_id="ses_test", task_id=task_id, agent="generation", action="Create",
    )


def _behave(monkeypatch: pytest.MonkeyPatch, **behavior: Any) -> None:
    monkeypatch.setenv("FAKE_CLI_BEHAVIOR", json.dumps(behavior))


# --------------------------------------------------------------------------- #
# Codex schema compatibility


def test_codex_output_schema_requires_all_declared_properties() -> None:
    schema = _strict_object_schema({
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "constraints": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
        "required": ["objective"],
    })

    assert schema["required"] == ["objective", "constraints"]
    assert schema["properties"]["constraints"]["required"] == ["name"]
    assert schema["additionalProperties"] is False


# --------------------------------------------------------------------------- #
# billing safety


def test_billing_env_vars_are_stripped() -> None:
    import os

    for var in BILLING_ENV_VARS:
        os.environ[var] = "leaked"
    try:
        env = sanitized_env()
        for var in BILLING_ENV_VARS:
            assert var not in env, f"{var} would make the CLI bill an API key"
    finally:
        for var in BILLING_ENV_VARS:
            os.environ.pop(var, None)


async def test_api_keys_are_stripped_from_the_subprocess_environment(
    provider: ClaudeCliProvider, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point of this backend: no call may be billed to an API key."""
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-reach-the-cli")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-reach-the-cli")
    monkeypatch.setenv("FAKE_CLI_DUMP", str(dump))
    _behave(monkeypatch, records=[["record_hypothesis", {"title": "T"}]])

    await provider.call(_spec(), _ctx())

    child_env = json.loads(dump.read_text())["env"]
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "OPENAI_API_KEY" not in child_env


# --------------------------------------------------------------------------- #
# command construction


def test_command_replaces_the_system_prompt_and_disables_builtin_tools(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    inv = provider.build_invocation(_spec(), _ctx(), tmp_path)

    assert "-p" in inv.argv
    assert inv.argv[inv.argv.index("--output-format") + 1] == "json"
    assert inv.argv[inv.argv.index("--model") + 1] == "opus"
    # Replacing rather than appending is what avoids ~19k tokens of harness
    # scaffolding on every call.
    assert "--system-prompt" in inv.argv
    assert "--append-system-prompt" not in inv.argv
    # Built-ins off so every tool call is recorded by our MCP server.
    assert inv.argv[inv.argv.index("--tools") + 1] == ""


def test_command_can_append_instead_of_replacing_the_system_prompt(
    provider: ClaudeCliProvider, cli_cfg: Config, tmp_path: Path
) -> None:
    cli_cfg.llm.claude_cli.replace_system_prompt = False

    inv = provider.build_invocation(_spec(), _ctx(), tmp_path)

    assert "--append-system-prompt" in inv.argv
    assert "--system-prompt" not in inv.argv


def test_mcp_config_wires_the_capture_dir_and_allowlists_only_our_tools(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    inv = provider.build_invocation(_spec(), _ctx(), tmp_path)

    assert "--strict-mcp-config" in inv.argv
    mcp = json.loads(inv.argv[inv.argv.index("--mcp-config") + 1])
    server = mcp["mcpServers"]["cosci"]
    assert server["env"]["COSCI_CAPTURE_DIR"] == str(tmp_path)
    assert server["env"]["COSCI_RECORD_TOOLS"] == "record_hypothesis"
    assert server["env"]["COSCI_AGENT"] == "generation"

    allowed = inv.argv[inv.argv.index("--allowedTools") + 1].split(",")
    assert set(allowed) == {"mcp__cosci__record_hypothesis", "mcp__cosci__pubmed_search"}


def test_calls_without_research_tools_expose_none_to_the_server(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    """Metareview and parse_goal must not be handed a search tool."""
    inv = provider.build_invocation(_spec(tools=[RECORD_TOOL]), _ctx(), tmp_path)

    mcp = json.loads(inv.argv[inv.argv.index("--mcp-config") + 1])
    assert mcp["mcpServers"]["cosci"]["env"]["COSCI_AGENT"] == ""


def test_prompt_goes_over_stdin_not_argv(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    """Meta-review prompts carry dozens of reviews; argv has a length limit."""
    big = "x" * 200_000
    inv = provider.build_invocation(
        _spec(user_blocks=[CachedBlock(big)]), _ctx(), tmp_path,
    )

    assert big in inv.stdin
    assert not any(big in arg for arg in inv.argv)


@pytest.mark.parametrize("tokens,expected", [
    (0, "low"), (1, "low"), (4_000, "medium"), (6_000, "high"),
    (8_000, "high"), (12_000, "xhigh"), (16_000, "xhigh"), (32_000, "max"),
])
def test_thinking_budgets_map_onto_effort_tiers(tokens: int, expected: str) -> None:
    assert effort_for_thinking_tokens(tokens) == expected


def test_forced_tool_choice_becomes_a_prompt_instruction() -> None:
    """Headless mode has no tool_choice flag, so the demand must be in words."""
    spec = _spec(tool_choice={"type": "tool", "name": "record_hypothesis"})

    prompt = build_prompt(spec)

    assert "record_hypothesis" in prompt
    assert "MUST" in prompt


def test_prompt_renders_a_prior_tool_thread() -> None:
    spec = _spec(extra_messages=[
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "pubmed_search", "input": {"q": "gut"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "content": "found 3 papers"},
        ]},
    ])

    prompt = build_prompt(spec)

    assert "pubmed_search" in prompt
    assert "found 3 papers" in prompt


# --------------------------------------------------------------------------- #
# output parsing


def test_parse_reads_usage_cost_and_turns(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    stdout = json.dumps({
        "is_error": False, "result": "hello", "num_turns": 4,
        "total_cost_usd": 0.25, "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10, "output_tokens": 20,
            "cache_creation_input_tokens": 30, "cache_read_input_tokens": 40,
        },
    })

    outcome = provider.parse_outcome(
        stdout=stdout, stderr="", returncode=0, capture_dir=tmp_path,
    )

    assert outcome.text == "hello"
    assert (outcome.input_tokens, outcome.output_tokens) == (10, 20)
    assert (outcome.cache_write, outcome.cache_read) == (30, 40)
    assert outcome.num_turns == 4
    assert outcome.reported_cost_usd == 0.25


def test_parse_tolerates_a_warning_line_before_the_json(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    stdout = 'Warning: no stdin data received in 3s\n{"is_error": false, "result": "ok"}'

    outcome = provider.parse_outcome(
        stdout=stdout, stderr="", returncode=0, capture_dir=tmp_path,
    )

    assert outcome.text == "ok"


def test_rate_limit_is_classified_retryable(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    stdout = json.dumps({
        "is_error": True, "subtype": "error_during_execution",
        "result": "Claude usage limit reached. Try again later.",
    })

    with pytest.raises(CliRetryableError):
        provider.parse_outcome(
            stdout=stdout, stderr="", returncode=0, capture_dir=tmp_path,
        )


def test_bad_model_is_classified_terminal(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    stdout = json.dumps({
        "is_error": True, "subtype": "error_during_execution",
        "result": "unknown model 'nope'",
    })

    with pytest.raises(CliBackendError) as e:
        provider.parse_outcome(
            stdout=stdout, stderr="", returncode=0, capture_dir=tmp_path,
        )
    assert not isinstance(e.value, CliRetryableError)


# --------------------------------------------------------------------------- #
# capture → response


def test_read_capture_collects_records_and_urls(tmp_path: Path) -> None:
    (tmp_path / "record-0001-record_hypothesis.json").write_text(
        json.dumps({"tool": "record_hypothesis", "input": {"title": "A"}})
    )
    (tmp_path / "tools.jsonl").write_text(
        json.dumps({"name": "pubmed_search", "args": {}, "is_error": False,
                    "duration_ms": 3, "urls": ["https://a.example"]}) + "\n"
    )

    bundle = read_capture(tmp_path)

    assert bundle.records == [("record_hypothesis", {"title": "A"})]
    assert bundle.seen_urls == {"https://a.example"}
    assert bundle.tool_calls[0]["name"] == "pubmed_search"


def test_read_capture_survives_a_truncated_line(tmp_path: Path) -> None:
    (tmp_path / "tools.jsonl").write_text('{"name": "a", "urls": []}\n{"name": tru')

    bundle = read_capture(tmp_path)

    assert [c["name"] for c in bundle.tool_calls] == ["a"]


def test_records_become_tool_use_blocks_agents_can_find() -> None:
    """`agents/base.py` locates structured output by scanning for tool_use."""
    from co_scientist.agents.base import BaseAgent
    from co_scientist.llm.cli_backend.base import CliOutcome
    from co_scientist.llm.types import LLMResponse

    outcome = CliOutcome(text="prose", input_tokens=1, output_tokens=2)
    capture = CaptureBundle(records=[("record_hypothesis", {"title": "T"})])

    message = build_message(_spec(), outcome, capture)
    response = LLMResponse(
        raw=message, transcript_id="t", cost_usd=0.0,
        input_tokens=1, output_tokens=2, cache_read=0, cache_write=0,
    )

    assert BaseAgent._final_tool_use(response, "record_hypothesis") == {"title": "T"}
    assert BaseAgent._final_text(response) == "prose"


# --------------------------------------------------------------------------- #
# end-to-end through the fake binary


async def test_call_roundtrip_records_a_transcript_and_returns_the_record(
    provider: ClaudeCliProvider, conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behave(
        monkeypatch,
        text="I found one.",
        records=[["record_hypothesis", {"title": "SCFA depletion"}]],
        tools=[{"name": "pubmed_search", "urls": ["https://pubmed.example/1"]}],
    )

    response = await provider.call(_spec(), _ctx())

    from co_scientist.agents.base import BaseAgent

    assert BaseAgent._final_tool_use(response, "record_hypothesis") == {
        "title": "SCFA depletion"
    }
    assert response.capture is not None
    assert response.capture.seen_urls == {"https://pubmed.example/1"}
    assert response.num_turns == 2
    assert response.cost_usd == pytest.approx(0.0178)

    async with conn.execute(
        "SELECT agent, model, input_tokens, cost_usd FROM transcripts"
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["agent"] == "generation"
    assert rows[0]["model"] == "opus"


async def test_failed_call_releases_its_budget_reservation(
    provider: ClaudeCliProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaked reservation would permanently shrink the agent's share."""
    _behave(monkeypatch, mode="error", text="unknown model 'nope'")

    for _ in range(5):
        with pytest.raises(CliBackendError):
            await provider.call(_spec(), _ctx())

    # If each failure had leaked its reservation, generation's share would be
    # exhausted by now and this call would be refused before it ever spawned.
    _behave(monkeypatch, records=[["record_hypothesis", {"title": "T"}]])
    response = await provider.call(_spec(), _ctx())

    assert response.capture is not None
    assert response.capture.records == [("record_hypothesis", {"title": "T"})]


async def test_unparseable_output_is_reported_not_swallowed(
    provider: ClaudeCliProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behave(monkeypatch, mode="garbage")

    with pytest.raises(CliBackendError):
        await provider.call(_spec(), _ctx())


async def test_a_lingering_grandchild_does_not_wedge_the_call(
    provider: ClaudeCliProvider, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the CLI spawns our MCP server, which inherits the stdout
    fd. If output were captured through a pipe, `communicate()` would wait for
    EOF — which cannot arrive while the grandchild holds that fd — and the
    call would hang until the timeout. Observed live as a ranking task sitting
    idle with no CLI process running.
    """
    import asyncio

    # A fake CLI that leaves a child holding stdout open after it exits.
    launcher = tmp_path / "fake-claude-with-orphan"
    launcher.write_text(
        "#!/bin/sh\n"
        "cat >/dev/null\n"                       # consume the prompt
        "sleep 30 &\n"                           # orphan inherits stdout
        'printf \'{"is_error":false,"result":"done","num_turns":1}\\n\'\n'
        "exit 0\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    provider._binary = str(launcher)
    provider._backend_cfg.timeout_seconds = 25

    # Must finish in about as long as the CLI itself takes, not the orphan's
    # lifetime and not the configured timeout.
    response = await asyncio.wait_for(provider.call(_spec(tools=[]), _ctx()), timeout=15)

    assert response.raw.content[0].text == "done"


async def test_a_hung_cli_is_killed_at_the_timeout(
    provider: ClaudeCliProvider, tmp_path: Path
) -> None:
    """A wedged CLI must not hold a worker forever — bound it and move on."""
    import asyncio

    launcher = tmp_path / "fake-claude-hang"
    launcher.write_text("#!/bin/sh\ncat >/dev/null\nsleep 60\n", encoding="utf-8")
    launcher.chmod(0o755)
    provider._binary = str(launcher)
    provider._backend_cfg.timeout_seconds = 1
    provider._retry.max_attempts = 1

    # Attempts are exhausted, so the retryable timeout surfaces wrapped.
    with pytest.raises(CliBackendError) as e:
        await asyncio.wait_for(provider.call(_spec(tools=[]), _ctx()), timeout=20)

    assert "timed out" in str(e.value)
