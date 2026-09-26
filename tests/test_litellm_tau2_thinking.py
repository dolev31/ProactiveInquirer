"""The tau2 proxy config never disabled the thinking channel. This is the regression test.

WHY THIS EXISTS. `conf/serving/litellm.yaml` (the Mac-side QA router) attaches
`extra_body.chat_template_kwargs.enable_thinking: false` to every `hosted_vllm/*` route,
because Qwen3 is a hybrid-reasoning model whose chat template emits a `<think>` channel by
default, and the rung-1 SFT rows were built with it OFF. `conf/serving/litellm.tau2.yaml` --
the SEPARATE config the tau2 campaign uses (see its own header comment: "SEPARATE FROM
`litellm.yaml` ON PURPOSE") -- never got the same treatment. Every tau2 unit that ran a local
Qwen3 questioner therefore ran with the thinking channel on, and 43.5% of prompted calls hit
the token cap silently truncated mid-thought (see memory: tau2-ran-with-qwen3-thinking-on).

THE IN-JOB GENERATED CONFIG IS NOT A SEPARATE CODE PATH. `scripts/hpc/tau2_cluster_campaign.sh`
builds its proxy config with a single `sed` substitution of the vLLM port against THIS SAME
FILE (`conf/serving/litellm.tau2.yaml`) -- it does not regenerate the model_list from scratch
and does not go through `router_add.py` (which only ever writes to `litellm.yaml` and refuses
outright on a file lacking its own insertion marker, which this file does not carry). So fixing
this file fixes the in-job config too, and the second test below proves that by running the
REAL sed line lifted out of the campaign script, not a reimplementation of it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
TAU2_CONFIG = REPO / "conf" / "serving" / "litellm.tau2.yaml"
CAMPAIGN_SCRIPT = REPO / "scripts" / "hpc" / "tau2_cluster_campaign.sh"
ROUTER_ADD = REPO / "scripts" / "hpc" / "router_add.py"


def _local_routes(doc: dict) -> list[dict]:
    """Every model_list entry whose litellm `model` is `hosted_vllm/*` -- i.e. served by a
    local vLLM process, as opposed to a gateway passthrough."""
    return [
        m
        for m in doc.get("model_list", [])
        if str(m.get("litellm_params", {}).get("model", "")).startswith("hosted_vllm/")
    ]


def _gateway_routes(doc: dict) -> list[dict]:
    return [m for m in doc.get("model_list", []) if m not in _local_routes(doc)]


def test_the_tau2_config_exists_and_parses():
    assert TAU2_CONFIG.is_file(), f"{TAU2_CONFIG} is missing"
    doc = yaml.safe_load(TAU2_CONFIG.read_text())
    assert doc.get("model_list"), "no model_list in the tau2 config"


def test_every_local_qwen_route_disables_thinking():
    """The defect, stated directly: every `hosted_vllm/*` route (all of which are Qwen3 in
    this file today) must set `extra_body.chat_template_kwargs.enable_thinking: False`,
    mirroring the form `conf/serving/litellm.yaml` already uses for the identical model ids.

    THIS TEST MUST FAIL AT THE UNCHANGED CODE. Before the fix, neither `qwen3-8b-base` nor
    `qwen3-8b-dpo-stacked-notdone-both` carries an `extra_body` key at all in
    `litellm.tau2.yaml`.
    """
    doc = yaml.safe_load(TAU2_CONFIG.read_text())
    local = _local_routes(doc)
    assert local, "expected at least one hosted_vllm/* route in the tau2 config"
    failures = []
    for m in local:
        params = m.get("litellm_params", {})
        kwargs = params.get("extra_body", {}).get("chat_template_kwargs", {})
        if kwargs.get("enable_thinking") is not False:
            failures.append((m.get("model_name"), params.get("model"), kwargs))
    assert not failures, (
        "these hosted_vllm/* routes do not disable the thinking channel "
        f"(model_name, litellm model, chat_template_kwargs found): {failures}"
    )


def test_gateway_passthrough_routes_are_unchanged():
    """The gateway passthrough routes (`openai/*`, `*`) must NOT gain a thinking kwarg: they
    are not Qwen3, `extra_body` there would be forwarded to a frozen role that never asked
    for it, and the task's own instruction is to touch nothing outside the existing
    hosted_vllm route blocks."""
    doc = yaml.safe_load(TAU2_CONFIG.read_text())
    gateway = _gateway_routes(doc)
    assert gateway, "expected at least one gateway passthrough route"
    names = {m.get("model_name") for m in gateway}
    assert names == {"openai/*", "*"}, names
    for m in gateway:
        assert "extra_body" not in m.get("litellm_params", {}), m
        assert m["litellm_params"]["api_base"] == "os.environ/PINQ_GATEWAY_BASE_URL"


def _extract_sed_generator(script_text: str) -> str:
    """Pull the literal `sed ... > "$CFG"` line the campaign uses to build its in-job proxy
    config out of the real script, so the test exercises the actual generator rather than a
    hand copy of it. Fails loudly if the script's shape has moved, rather than silently
    testing nothing (the same failure class `router_add.py`'s own docstring warns about)."""
    m = re.search(
        r'^sed "(?P<expr>s\|[^"]+)" "\$WT/conf/serving/litellm\.tau2\.yaml" > "\$CFG"$',
        script_text,
        re.MULTILINE,
    )
    assert m, (
        'could not find the expected `sed ... conf/serving/litellm.tau2.yaml > "$CFG"` line '
        f"in {CAMPAIGN_SCRIPT} -- the generator's shape has changed and this test no longer "
        "exercises the real code path"
    )
    return m.group("expr")


def test_the_injob_generated_config_still_disables_thinking(tmp_path):
    """Run the REAL sed line from `tau2_cluster_campaign.sh` against the committed config and
    parse what it actually produces -- the same object the in-job litellm proxy is launched
    with. This is not a re-implementation: it is the script's own substitution, extracted and
    executed, standing in for the campaign's `$CFG` temp file.
    """
    assert CAMPAIGN_SCRIPT.is_file(), f"{CAMPAIGN_SCRIPT} is missing"
    script_text = CAMPAIGN_SCRIPT.read_text()
    sed_expr = _extract_sed_generator(script_text)
    # The real script substitutes a caller-chosen $VPORT for the literal placeholder port
    # (8402); pin a concrete port here the way a real launch would.
    port = "19999"
    resolved_expr = sed_expr.replace("$VPORT", port)
    out = tmp_path / "litellm.tau2.generated.yaml"
    proc = subprocess.run(
        ["sed", resolved_expr, str(TAU2_CONFIG)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out.write_text(proc.stdout)
    assert f"127.0.0.1:{port}" in out.read_text(), "port substitution produced no new route"
    assert "127.0.0.1:8402" not in out.read_text(), "the placeholder port survived substitution"

    doc = yaml.safe_load(out.read_text())
    local = _local_routes(doc)
    assert local, "the generated config has no hosted_vllm/* route"
    failures = []
    for m in local:
        params = m.get("litellm_params", {})
        kwargs = params.get("extra_body", {}).get("chat_template_kwargs", {})
        if kwargs.get("enable_thinking") is not False:
            failures.append((m.get("model_name"), kwargs))
    assert not failures, (
        "the IN-JOB GENERATED config (real sed substitution of the committed file) still has "
        f"routes with the thinking channel not disabled: {failures}"
    )


def test_router_add_cannot_silently_add_a_route_to_the_tau2_config(tmp_path):
    """`router_add.py` is the QA proxy's own generator, and its `enable_thinking` logic
    (`wants_thinking_kwarg`) is the reference form this fix mirrors. It is NOT a code path
    that writes to `litellm.tau2.yaml`: that file carries no insertion marker
    (`# >>> router_add.py inserts local deployments above this line <<<`), so `add()` refuses
    outright rather than silently appending a route with no thinking guarantee at all. This
    pins that refusal so a future change that gives the tau2 config a marker (and thereby a
    silent, un-reviewed route-add path) is caught here rather than discovered in a campaign.

    Operates on a COPY, never the committed file: the assertion under test is that `add()`
    raises before writing, and running it against a tmp copy means a regression that makes it
    write anyway corrupts a throwaway file instead of this repo's tracked config.
    """
    import importlib.util
    import shutil

    spec = importlib.util.spec_from_file_location("router_add", ROUTER_ADD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert mod.MARKER not in TAU2_CONFIG.read_text(), (
        "litellm.tau2.yaml now carries router_add.py's insertion marker -- it can be written "
        "to by router_add.add() and this test (plus the manual review it stands in for) must "
        "be revisited to cover that path"
    )
    copy = tmp_path / "litellm.tau2.yaml"
    shutil.copy2(TAU2_CONFIG, copy)
    with pytest.raises(ValueError, match="insertion marker"):
        mod.add(copy, ["qwen3-8b-base"], port=8402)


# --------------------------------------------------------------------- the s1/s2 rerun names
#
# THE DEFECT THIS SECTION PINS. The rerun pins `qwen3-8b-dpo-stacked-notdone-both-s1` and `-s2`,
# neither of which appeared in `litellm.tau2.yaml`'s `model_list`. A model name with NO exact
# `model_name` entry does not error -- litellm's own routing falls through to the bare `"*"`
# pattern, rewrites it through the `openai/*` template, and forwards it to the REMOTE GATEWAY,
# which answers 200 from whatever model the gateway maps that (nonsensical) name to. A 200 with
# a completion is exactly what `thinking_probe.py`'s route proof and the campaign's own route
# proof both accept -- neither one can tell "the right local model answered" from "a different
# model on a different host answered", because both look identical on the wire: 200, a message,
# some tokens. This is not hypothetical: it is the literal shape of a silently wrong published
# number, which is why CLAUDE.md's rule 1 requires a `run_id` to name what actually answered.
#
# THE TEST USES LITELLM'S OWN `Router`, NOT A REIMPLEMENTATION OF ITS MATCHING RULES.
# `Router.get_model_list(model_name=...)` is the exact call the proxy's request path uses to
# decide what a name resolves to: an EXACT `model_name` match first (`_get_all_deployments`),
# and ONLY if that returns nothing does it fall through to `self.pattern_router.route(...)` --
# see the method's own source, reproduced in the module docstring below for anyone auditing
# this without a debugger. Building the real `Router` from the file's own `model_list` and
# blocking the socket layer proves two things at once: the resolution is litellm's real code
# path, and building it makes no network call (`Router.__init__` needs none for a static
# `model_list` with no health checks configured).
RERUN_MODELS = (
    "qwen3-8b-base",
    "qwen3-8b-dpo-stacked-notdone-both-s1",
    "qwen3-8b-dpo-stacked-notdone-both-s2",
)


class _NetworkBlocked(RuntimeError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Building a `litellm.Router` from a static `model_list` must never open a socket. If a
    future litellm version adds an implicit health check or cost-map fetch to `__init__` or to
    `get_model_list`, this fixture turns that into a loud test failure instead of a silent
    real network call from a test."""
    import socket

    def _blocked(*_a, **_kw):
        raise _NetworkBlocked("a test that resolves routes attempted a real socket connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


def _router_from(config_path: Path):
    router_mod = pytest.importorskip("litellm.router")
    doc = yaml.safe_load(config_path.read_text())
    return router_mod.Router(model_list=doc["model_list"])


def test_every_rerun_model_resolves_to_an_explicit_hosted_vllm_route(no_network):
    """(a) None of the three names the rerun pins may fall through to a wildcard. A wildcard
    hit is silently indistinguishable from a correct answer -- both are a 200 with a
    completion -- so this has to be proven against litellm's actual matching order, not
    asserted from the response shape.

    THIS TEST MUST FAIL AT THE UNCHANGED CODE: `-s1` and `-s2` have no `model_name` entry, so
    each resolves through the bare `"*"` -> `openai/*` gateway pattern instead.
    """
    router = _router_from(TAU2_CONFIG)
    failures = []
    for name in RERUN_MODELS:
        resolved = router.get_model_list(model_name=name)
        models = {d["litellm_params"]["model"] for d in (resolved or [])}
        if models != {f"hosted_vllm/{name}"}:
            failures.append((name, sorted(models)))
    assert not failures, (
        "these rerun model names did not resolve to an explicit hosted_vllm/* route "
        f"(name, resolved litellm model(s) instead): {failures}"
    )


def test_all_three_rerun_models_resolve_to_the_same_api_base(no_network):
    """(b) base, s1 and s2 all reach ONE serve (LSF 910128) behind ONE tunnel (8402 ->
    gpu-n595:8302). A route resolving to a different api_base would silently reach a
    different backend even while still naming the right model.
    """
    router = _router_from(TAU2_CONFIG)
    api_bases = {}
    for name in RERUN_MODELS:
        resolved = router.get_model_list(model_name=name)
        assert resolved, f"{name} did not resolve to anything"
        bases = {d["litellm_params"].get("api_base") for d in resolved}
        assert len(bases) == 1, f"{name} resolved to more than one api_base: {bases}"
        api_bases[name] = next(iter(bases))
    assert len(set(api_bases.values())) == 1, (
        f"the rerun's three models do not share one api_base: {api_bases}"
    )
    assert api_bases[RERUN_MODELS[0]] == "http://127.0.0.1:8402/v1", api_bases
