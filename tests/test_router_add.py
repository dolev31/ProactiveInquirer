"""Adding a served name to the Mac-side router must be mechanical.

WHY. `conf/serving/litellm.yaml` is the only way a sweep reaches two endpoints at once
(`LiteLLMClient` holds ONE base url for every role), so every checkpoint served on the
cluster needs a `model_list` entry here before `PI_MODEL_INQUIRER` can name it. There are
seventeen more checkpoints coming. Each entry is six lines, five of which are identical
between entries and one of which -- `enable_thinking: false` -- silently changes the PROMPT
if it is forgotten: the rung-1 rows were built with the thinking channel off, so a served
prompt with it on is a different prompt than the one the adapter was fitted to, and the dev
NLL that selected the checkpoint no longer describes what the policy sees.

WHY THE FILE IS EDITED AS TEXT AND NOT ROUND-TRIPPED THROUGH PyYAML. Two thirds of this file
is the reasoning for its own shape -- why the gateway url has its own variable name, why
`extra_body` is per-entry, how to launch the proxy. `yaml.safe_dump` would delete all of it.
The script inserts before a marker line and touches nothing else, and
`test_the_design_notes_survive` is what keeps that true.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hpc" / "router_add.py"
ROUTER = REPO / "conf" / "serving" / "litellm.yaml"


def _mod():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist")
    spec = importlib.util.spec_from_file_location("router_add", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def router(tmp_path):
    p = tmp_path / "litellm.yaml"
    shutil.copy2(ROUTER, p)
    return p


def _names(path: Path) -> list[str]:
    return [m["model_name"] for m in yaml.safe_load(path.read_text())["model_list"]]


def _entry(path: Path, name: str) -> dict:
    for m in yaml.safe_load(path.read_text())["model_list"]:
        if m["model_name"] == name:
            return m
    raise AssertionError(f"{name} not in {_names(path)}")


# --------------------------------------------------------------------------- the shipped file


def test_the_shipped_router_parses_and_ends_with_the_wildcard():
    """The wildcard is the passthrough to the team gateway. It must stay LAST: a catch-all
    ahead of an exact name shadows it on some litellm versions, and every local checkpoint
    would then be dispatched at the gateway under a name it has never heard of."""
    doc = yaml.safe_load(ROUTER.read_text())
    assert _names(ROUTER)[-1] == "*"
    assert doc["model_list"][-1]["litellm_params"]["api_base"] == "os.environ/PINQ_GATEWAY_BASE_URL"
    # Pinned 2026-09-18 (lane L0.8): the tail is exactly the two bare "*" entries, in that
    # position. The "openai/*" identity-passthrough pair (see below) must sit BEFORE them --
    # never after, or it would itself be shadowed by the bare wildcard on some litellm
    # versions, the same failure mode the marker comment above the wildcard warns about.
    assert _names(ROUTER)[-2:] == ["*", "*"]


def test_the_wildcard_is_sharded_across_both_gateway_keys():
    """Lane L0.5 (2026-09-18). `.env` holds three key-shaped vars but only two DISTINCT keys:
    `LITELLM_API_KEY` and `LITELLM_API_KEY_2` are byte-identical (measured 2026-09-17) and
    share one quota; `PI_AGENT_LITELLM_API_KEY` is the only other distinct credential. A sweep
    naming one gateway model should draw on both teams' rate limits, not one. litellm's
    `PatternMatchRouter.add_pattern` stores every deployment registered under one pattern as a
    LIST, and `_common_checks_available_deployment` hands that whole list to the router's
    normal deployment selection -- so the wildcard passthrough must be registered TWICE, each
    copy differing only in `api_key`, for the router to load-balance across both keys instead
    of only ever dispatching the first.  `router_add.py` cannot generate this entry: `_NAME`
    refuses "*" and `entry_yaml` hardcodes `hosted_vllm/<name>` plus a 127.0.0.1 `api_base`,
    so this pair is maintained by hand."""
    doc = yaml.safe_load(ROUTER.read_text())
    wildcards = [m for m in doc["model_list"] if m["model_name"] == "*"]
    assert len(wildcards) == 2, wildcards
    keys = {w["litellm_params"]["api_key"] for w in wildcards}
    assert keys == {"os.environ/LITELLM_API_KEY", "os.environ/PI_AGENT_LITELLM_API_KEY"}
    for w in wildcards:
        assert w["litellm_params"]["model"] == "openai/*"
        assert w["litellm_params"]["api_base"] == "os.environ/PINQ_GATEWAY_BASE_URL"
    # both copies stay LAST: a catch-all ahead of an exact local name shadows it on some
    # litellm versions (see the marker comment above them).
    assert doc["model_list"][-2:] == wildcards


def test_the_prefixed_wildcard_is_also_sharded_across_both_gateway_keys():
    """Lane L0.8 (2026-09-18). Measured: a raw request whose model is already prefixed
    (`openai/aws/gpt-oss-120b`, i.e. a role pin copied verbatim instead of going through
    `MeteredClient`, which strips `openai/` client-side) used to match the bare `"*"` entry,
    which captures the FULL string and re-prefixes it -- `openai/openai/aws/gpt-oss-120b` --
    one `openai/` survives the wire strip and the gateway refuses a name on no team's
    allow-list. This is the exact shape of the 22:07 Drafter refusal. `openai/*` is a more
    specific litellm pattern than `*` (longer pattern wins, see
    `PatternUtils.calculate_pattern_specificity`), so registering it explicitly intercepts an
    already-prefixed request before the bare wildcard ever sees it. It needs the same
    two-key sharding as the bare wildcard, for the same reason."""
    doc = yaml.safe_load(ROUTER.read_text())
    prefixed = [m for m in doc["model_list"] if m["model_name"] == "openai/*"]
    assert len(prefixed) == 2, prefixed
    keys = {w["litellm_params"]["api_key"] for w in prefixed}
    assert keys == {"os.environ/LITELLM_API_KEY", "os.environ/PI_AGENT_LITELLM_API_KEY"}
    for w in prefixed:
        assert w["litellm_params"]["model"] == "openai/*"
        assert w["litellm_params"]["api_base"] == "os.environ/PINQ_GATEWAY_BASE_URL"
    # immediately precedes the bare "*" pair -- both must stay at the very end (see
    # test_the_shipped_router_parses_and_ends_with_the_wildcard), and the more specific
    # pattern is useless if a less specific one were somehow consulted first.
    names = _names(ROUTER)
    assert names[-4:] == ["openai/*", "openai/*", "*", "*"]


def test_an_already_prefixed_role_pin_is_not_double_prefixed():
    """The regression test for the 22:07 refusal. Exercises litellm's OWN
    `PatternMatchRouter` against the shipped file's wildcard entries -- no mock -- so this
    fails if a future edit changes pattern specificity, template shape, or drops the
    `openai/*` pair and reintroduces the double prefix.

    Before the fix (only the bare `"*"` -> `openai/*` pair existed) this test failed:
    routing `openai/aws/gpt-oss-120b` produced `openai/openai/aws/gpt-oss-120b`.
    """
    pmr_mod = pytest.importorskip("litellm.router_utils.pattern_match_deployments")
    router = pmr_mod.PatternMatchRouter()
    doc = yaml.safe_load(ROUTER.read_text())
    for entry in doc["model_list"]:
        if entry["model_name"] in ("*", "openai/*"):
            router.add_pattern(entry["model_name"], entry)

    # A raw request that still carries the role pin's "openai/" prefix verbatim (what a
    # hand-built curl smoke test sends) must round-trip to itself -- one prefix, not two.
    already_prefixed = router.route("openai/aws/gpt-oss-120b")
    assert already_prefixed, "no pattern matched an already-prefixed request"
    models = {d["litellm_params"]["model"] for d in already_prefixed}
    assert models == {"openai/aws/gpt-oss-120b"}
    keys = {d["litellm_params"]["api_key"] for d in already_prefixed}
    assert keys == {"os.environ/LITELLM_API_KEY", "os.environ/PI_AGENT_LITELLM_API_KEY"}

    # The normal sweep path -- the SDK already stripped "openai/" before the request left the
    # Mac -- must be unaffected: the bare wildcard still adds exactly one prefix.
    bare = router.route("aws/gpt-oss-120b")
    assert bare, "no pattern matched a bare (already-stripped) request"
    bare_models = {d["litellm_params"]["model"] for d in bare}
    assert bare_models == {"openai/aws/gpt-oss-120b"}


def test_the_shipped_router_carries_the_insertion_marker():
    """Without it the script has nowhere to insert, and an entry appended at the end of the
    file would sit AFTER the wildcard."""
    ra = _mod()
    assert ra.MARKER in ROUTER.read_text()


def test_no_secret_is_in_the_shipped_router():
    """The gateway key is read from the environment at proxy start and never written here."""
    text = ROUTER.read_text()
    assert "sk-" not in text
    assert "os.environ/LITELLM_API_KEY" in text


# --------------------------------------------------------------------------- the generator


def test_the_generated_entry_reproduces_the_hand_written_one(router):
    """The strongest check available: `qwen3-8b-sft` was written by hand and MEASURED to work
    (19 prompt tokens through the proxy, matching the trainer's rendering). If the generator
    produces anything else for that name, the generator is wrong."""
    ra = _mod()
    generated = yaml.safe_load(ra.entry_yaml("qwen3-8b-sft", port=8000))[0]
    assert generated == _entry(ROUTER, "qwen3-8b-sft")


def test_a_qwen_name_carries_enable_thinking_false(router):
    ra = _mod()
    ra.add(router, ["qwen3-8b-sft-headline"], port=8000)
    e = _entry(router, "qwen3-8b-sft-headline")
    assert e["litellm_params"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert e["litellm_params"]["model"] == "hosted_vllm/qwen3-8b-sft-headline"


def test_a_non_qwen_name_carries_no_chat_template_kwargs(router):
    """`extra_body` is per-client and `enable_thinking` is a Qwen chat-template kwarg. On a
    model whose template does not take it, a served request would carry an argument the
    template ignores at best and rejects at worst."""
    ra = _mod()
    ra.add(router, ["llama3-8b-sft"], port=8000)
    assert "extra_body" not in _entry(router, "llama3-8b-sft")["litellm_params"]


def test_the_port_reaches_api_base(router):
    """One vLLM instance per base model: the 4B and 1.7B live on their own ports."""
    ra = _mod()
    ra.add(router, ["qwen3-4b-sft-headline"], port=8001)
    assert (
        _entry(router, "qwen3-4b-sft-headline")["litellm_params"]["api_base"]
        == "http://127.0.0.1:8001/v1"
    )


# --------------------------------------------------------------------------- the edit


def test_the_new_entry_lands_before_the_wildcard(router):
    ra = _mod()
    ra.add(router, ["qwen3-8b-dpo-pooled-control"], port=8000)
    names = _names(router)
    assert names[-1] == "*"
    assert names.index("qwen3-8b-dpo-pooled-control") < names.index("*")


def test_several_names_at_once_keep_their_order(router):
    ra = _mod()
    ra.add(router, ["a-qwen3-x", "b-qwen3-y", "c-qwen3-z"], port=8000)
    names = _names(router)
    assert names.index("a-qwen3-x") < names.index("b-qwen3-y") < names.index("c-qwen3-z")


def test_it_is_idempotent(router):
    """It will be run again: over a loop of finished runs, after an interrupted edit."""
    ra = _mod()
    ra.add(router, ["qwen3-8b-sft-headline"], port=8000)
    once = router.read_text()
    ra.add(router, ["qwen3-8b-sft-headline"], port=8000)
    assert router.read_text() == once
    assert _names(router).count("qwen3-8b-sft-headline") == 1


def test_adding_a_name_that_is_already_there_leaves_the_file_untouched(router):
    ra = _mod()
    before = router.read_text()
    ra.add(router, ["qwen3-8b-sft"], port=8000)
    assert router.read_text() == before


def test_the_design_notes_survive(router):
    """The file is mostly its own reasoning. A YAML round-trip would delete every line of it."""
    ra = _mod()
    ra.add(router, ["qwen3-8b-sft-headline"], port=8000)
    text = router.read_text()
    assert "WHY THIS FILE HAS TO EXIST" in text
    assert "NO SECRET IS IN THIS FILE" in text
    assert "WHY `enable_thinking: false`" in text
    assert "PINQ_GATEWAY_BASE_URL" in text


def test_the_result_still_parses_and_holds_no_secret(router):
    ra = _mod()
    ra.add(router, ["qwen3-8b-sft-headline", "qwen3-8b-dpo-pooled-rater"], port=8000)
    doc = yaml.safe_load(router.read_text())
    assert doc["litellm_settings"]["num_retries"] == 0
    assert doc["general_settings"]["disable_spend_logs"] is True
    assert "sk-" not in router.read_text()


def test_a_file_without_the_marker_is_refused(tmp_path):
    """Refusing beats appending: an entry after the wildcard is dead configuration that looks
    alive."""
    ra = _mod()
    p = tmp_path / "bare.yaml"
    p.write_text("model_list:\n  - model_name: x\n")
    with pytest.raises(ValueError, match="marker"):
        ra.add(p, ["qwen3-8b-sft-headline"], port=8000)


@pytest.mark.parametrize("bad", ["a b", "a/b", "", "*"])
def test_a_name_that_is_not_a_served_id_is_refused(router, bad):
    ra = _mod()
    with pytest.raises(ValueError):
        ra.add(router, [bad], port=8000)


def test_the_cli_edits_the_file_and_reports_what_it_did(router, capsys):
    ra = _mod()
    assert (
        ra.main(["--name", "qwen3-8b-sft-headline", "--port", "8000", "--router", str(router)]) == 0
    )
    assert "qwen3-8b-sft-headline" in capsys.readouterr().out
    assert "qwen3-8b-sft-headline" in _names(router)


def test_the_cli_returns_nonzero_when_it_refuses(tmp_path, capsys):
    ra = _mod()
    p = tmp_path / "bare.yaml"
    p.write_text("model_list: []\n")
    assert ra.main(["--name", "x-qwen3", "--router", str(p)]) != 0


def test_re_adding_a_name_with_another_port_moves_its_api_base(router):
    """MEASURED 2026-09-15: `router_add.py --name qwen3-8b-sft-headline --port 8001` printed
    "already present -> http://127.0.0.1:8001/v1" and left the entry on 8000 -- the port it was
    asked for, not the one stored -- so every rollout for that name would have gone to the
    server that does not hold the adapter. A second vLLM instance is the normal case (one per
    base, one per wave), so an existing name must follow the port it is given."""
    ra = _mod()
    ra.add(router, ["qwen3-8b-sft-headline"], port=8000)
    moved = ra.add(router, ["qwen3-8b-sft-headline"], port=8001)
    text = router.read_text()
    assert moved == ["qwen3-8b-sft-headline"]
    assert _names(router).count("qwen3-8b-sft-headline") == 1
    block = text[text.index("  - model_name: qwen3-8b-sft-headline") :]
    block = block[: block.index("  - model_name:", 1)] if "  - model_name:" in block[1:] else block
    assert "http://127.0.0.1:8001/v1" in block and "http://127.0.0.1:8000/v1" not in block
    # and the same port again is a no-op
    before = router.read_text()
    assert ra.add(router, ["qwen3-8b-sft-headline"], port=8001) == []
    assert router.read_text() == before


# ------------------------------------------- a comment BETWEEN TWO KEYS inside the mapping

COMMENT_IN_MAPPING = (
    "      # this one is the 8B adapter, so it follows the 8B instance; the 4B trio below\n"
    "      # is on :8001 and an adapter cannot be applied to a base it was not trained on.\n"
)


def _raw_entry(text: str, name: str) -> str:
    """The raw text of one `- model_name: <name>` item, comments included.

    An item runs from its `- model_name:` line to the next non-blank line indented less than
    its own keys -- the next `  - ` item, the `  #` marker, or a top-level key. `yaml.safe_load`
    cannot be used to inspect this: the defect below is INVISIBLE after parsing, because a
    duplicate key inside one mapping is silently resolved to the last one.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.rstrip("\n") == f"  - model_name: {name}")
    end = start + 1
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith("    "):
            break
        end = i + 1
    return "".join(lines[start:end])


@pytest.fixture
def commented_router(tmp_path):
    """The shipped file, annotated the way an operator annotates it: a note between two keys
    INSIDE `litellm_params`. It is valid YAML, it survives `yaml.safe_load`, and it is the
    shape `qwen3-8b-sft-headline` was found in at 15:40 IDT on 2026-09-15."""
    text = ROUTER.read_text()
    anchor = "      model: hosted_vllm/qwen3-8b-sft-headline\n"
    assert text.count(anchor) == 1, "the fixture anchor is no longer unique in the shipped file"
    p = tmp_path / "litellm.yaml"
    p.write_text(text.replace(anchor, anchor + COMMENT_IN_MAPPING, 1))
    assert yaml.safe_load(p.read_text())["model_list"], "the fixture itself must be valid YAML"
    return p


def test_moving_the_port_past_a_comment_leaves_exactly_one_api_base(commented_router):
    """MEASURED 2026-09-15 15:40 IDT: `router_add.py --name qwen3-8b-sft-headline --port ...`
    printed "added or moved to ..." and the effective port did not move. The entry carried a
    comment between `model` and `api_base`; the replacement stopped AT that comment, so a
    fresh `api_base`/`api_key`/`extra_body` block was written above it while the original keys
    stayed below. One mapping then held two `api_base` keys, `yaml.safe_load` kept the LAST --
    the old one -- and the router went on dispatching that name to the instance that does not
    hold its adapter, with no parse error anywhere. Both halves are the test: what the loader
    sees AND that the raw text holds one `api_base:` line. The port alone would have passed
    the moment the duplicate happened to be ordered the other way."""
    ra = _mod()
    ra.add(commented_router, ["qwen3-8b-sft-headline"], port=8001)
    raw = _raw_entry(commented_router.read_text(), "qwen3-8b-sft-headline")
    assert raw.count("api_base:") == 1, raw
    assert raw.count("api_key:") == 1, raw
    e = _entry(commented_router, "qwen3-8b-sft-headline")
    assert e["litellm_params"]["api_base"] == "http://127.0.0.1:8001/v1"


def test_the_operator_note_inside_the_entry_is_not_deleted(commented_router):
    """DELIBERATE: the item is rewritten whole, and a comment that was inside it is re-emitted
    at the end of the item rather than dropped. Two thirds of this file is the reasoning for
    its own shape; a tool that silently deletes a line of that reasoning is worse than one that
    moves it, and a moved line is visible in `git diff` while a deleted one is not. The comment
    can no longer sit against the key it annotated, because that key has been rewritten."""
    ra = _mod()
    ra.add(commented_router, ["qwen3-8b-sft-headline"], port=8001)
    raw = _raw_entry(commented_router.read_text(), "qwen3-8b-sft-headline")
    assert "an adapter cannot be applied to a base it was not trained on" in raw


def test_moving_the_port_past_a_comment_is_still_idempotent(commented_router):
    ra = _mod()
    ra.add(commented_router, ["qwen3-8b-sft-headline"], port=8001)
    once = commented_router.read_text()
    assert ra.add(commented_router, ["qwen3-8b-sft-headline"], port=8001) == []
    assert commented_router.read_text() == once


# --------------------------------------------------------------- the duplicate-key guard


def test_a_duplicate_key_in_one_mapping_is_refused_before_anything_is_written(tmp_path):
    """A duplicate key is not a parse error. `yaml.safe_load` keeps the LAST one and says
    nothing, so the file reads as valid while the router serves a port nobody asked for --
    exactly the failure above, one layer down. Only a raw-text scan can see it, so the write
    is gated on one, and the gate fires BEFORE the file is touched."""
    ra = _mod()
    p = tmp_path / "dupe.yaml"
    p.write_text(
        "model_list:\n"
        "  - model_name: qwen3-8b-sft-headline\n"
        "    litellm_params:\n"
        "      model: hosted_vllm/qwen3-8b-sft-headline\n"
        "      api_base: http://127.0.0.1:8001/v1\n"
        "      # the note the second one hid behind\n"
        "      api_base: http://127.0.0.1:8000/v1\n"
        '      api_key: "not-used-vllm-is-open"\n'
        f"  {ra.MARKER}\n"
    )
    before = p.read_text()
    loaded = yaml.safe_load(before)["model_list"][0]["litellm_params"]["api_base"]
    assert loaded == "http://127.0.0.1:8000/v1", "the loader hides it; that is the whole point"
    with pytest.raises(ValueError, match="duplicate key"):
        ra.add(p, ["qwen3-4b-sft-headline"], port=8001)
    assert p.read_text() == before, "a refused write must not half-write"


def test_the_duplicate_key_scan_names_the_key_and_both_lines():
    ra = _mod()
    dupes = ra.duplicate_keys(
        "model_list:\n"
        "  - model_name: a\n"
        "    litellm_params:\n"
        "      api_base: one\n"
        "      api_base: two\n"
    )
    assert dupes == [(5, 4, "api_base")]


def test_the_same_key_in_two_different_mappings_is_not_a_duplicate():
    """Every entry in the file carries `model:`, `api_base:` and `api_key:`. A scan that
    cannot tell two mappings apart would refuse every real file and be turned off on day one."""
    ra = _mod()
    assert ra.duplicate_keys(ROUTER.read_text()) == []


def test_the_shipped_router_holds_no_duplicate_key():
    """The live file the proxy actually loads. If this fails, a served name is on a port that
    no printed line ever named."""
    ra = _mod()
    assert ra.duplicate_keys(ROUTER.read_text()) == []


def test_the_cli_refuses_a_duplicate_key_file_with_a_nonzero_status(tmp_path, capsys):
    ra = _mod()
    p = tmp_path / "dupe.yaml"
    p.write_text(
        "model_list:\n"
        "  - model_name: x-qwen3\n"
        "    litellm_params:\n"
        "      api_base: one\n"
        "      api_base: two\n"
        f"  {ra.MARKER}\n"
    )
    assert ra.main(["--name", "y-qwen3", "--router", str(p)]) != 0
    assert "duplicate key" in capsys.readouterr().err


def test_an_entry_already_broken_by_the_old_edit_is_repaired_by_rewriting_it(tmp_path):
    """The file as it was FOUND at 15:40: one mapping holding both ports. The guard must not
    strand the operator in it -- re-running for that name rebuilds the item from one rendering,
    so the duplicate is gone and the write goes through."""
    ra = _mod()
    p = tmp_path / "broken.yaml"
    p.write_text(
        "model_list:\n"
        "  - model_name: qwen3-8b-sft-headline\n"
        "    litellm_params:\n"
        "      model: hosted_vllm/qwen3-8b-sft-headline\n"
        "      api_base: http://127.0.0.1:8000/v1\n"
        '      api_key: "not-used-vllm-is-open"\n'
        "      # the comment the first block was written above\n"
        "      api_base: http://127.0.0.1:8001/v1\n"
        '      api_key: "not-used-vllm-is-open"\n'
        f"  {ra.MARKER}\n"
    )
    assert ra.duplicate_keys(p.read_text())
    assert ra.add(p, ["qwen3-8b-sft-headline"], port=8000) == ["qwen3-8b-sft-headline"]
    assert ra.duplicate_keys(p.read_text()) == []
    assert _entry(p, "qwen3-8b-sft-headline")["litellm_params"]["api_base"] == (
        "http://127.0.0.1:8000/v1"
    )
