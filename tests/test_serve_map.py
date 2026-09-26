"""One vLLM process must be able to serve MANY checkpoints, from a map nobody hand-edits.

WHAT WAS WRONG. `scripts/hpc/serve.sh` carried the adapter map as two parallel bash arrays
(`DIRS=(rung1-8b rung2-8b-control ...)`, `NAMES=(qwen3-8b-sft ...)`), so serving a new
checkpoint meant editing a committed script -- and a committed edit on the cluster is a dirty
tree, which `hpc_banner` refuses to start under. Worse, a directory that was absent was
SKIPPED with a log line: a sweep pointed at `qwen3-8b-dpo-rater` would 404 at request time
rather than fail at launch, and the four-hour queue wait would already have been spent.

WHAT REPLACES IT. `PINQ_LORA="name=path[,name=path...]"` (or `PINQ_LORA_FILE`), parsed here,
in Python, where it can be tested without a GPU. A path that is not an adapter is a REFUSAL,
not a skip. The served set, with the sha256 of every weights file it loaded, is written to
`$ART/serve/SERVED.<jobid>.json` so a sweep can prove afterwards which weights answered it --
`conf/checkpoints.json` says what SHOULD be served; this file says what WAS.

`--max-loras` is the size of the map (vLLM refuses to load more adapters at once than this),
`--max-lora-rank 32` is what every rung-1/2 adapter is trained at (`lora.r` 32 in
rung1.manifest.json), and `--max-model-len` defaults to 16384: measured to fit on the 80 GB
card with the 8B at 16.31 GiB of weights and 54.46 GiB of KV cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hpc" / "serve_map.py"
SERVE_SH = REPO / "scripts" / "hpc" / "serve.sh"


def _mod():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist")
    spec = importlib.util.spec_from_file_location("serve_map", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _adapter(root: Path, name: str, *, weights: bytes = b"w", config: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(weights)
    if config:
        (d / "adapter_config.json").write_text(json.dumps({"r": 32}))
    return d


# --------------------------------------------------------------------------- parsing


def test_parse_spec_reads_name_equals_path_pairs_in_order():
    sm = _mod()
    assert sm.parse_spec("a=x/1,b=y/2") == [("a", "x/1"), ("b", "y/2")]


def test_parse_spec_tolerates_whitespace_and_a_trailing_comma():
    sm = _mod()
    assert sm.parse_spec(" a = x/1 , b=y/2 , ") == [("a", "x/1"), ("b", "y/2")]


def test_an_empty_spec_is_an_empty_map_not_an_error():
    """A base-only serve is legitimate -- it is what the box does before any rung finishes."""
    sm = _mod()
    assert sm.parse_spec("") == []
    assert sm.parse_spec("   \n  ") == []


def test_parse_spec_refuses_a_segment_without_an_equals():
    """A bare path would be served under no name, and every request for it would 404."""
    sm = _mod()
    with pytest.raises(ValueError, match="artifacts/rung1-8b"):
        sm.parse_spec("artifacts/rung1-8b")


def test_parse_spec_refuses_a_duplicate_name():
    """vLLM would keep one of them silently, and the price table, the checkpoint registry and
    `PI_MODEL_INQUIRER` are all keyed on this exact string -- so the sweep would be priced and
    provenanced against weights it never called."""
    sm = _mod()
    with pytest.raises(ValueError, match="qwen3-8b-sft"):
        sm.parse_spec("qwen3-8b-sft=a,qwen3-8b-sft=b")


@pytest.mark.parametrize("bad", ["a b=p", "a/b=p", "=p", "a=", 'a"=p'])
def test_parse_spec_refuses_a_name_that_is_not_a_served_id(bad):
    """The name goes on the wire as an OpenAI model id and into a shell word."""
    sm = _mod()
    with pytest.raises(ValueError):
        sm.parse_spec(bad)


def test_the_map_may_come_from_a_file_with_comments(tmp_path):
    """`PINQ_LORA_FILE`: ten adapters on one bsub line is unreadable, and a file can be
    reviewed. Newlines separate exactly as commas do."""
    sm = _mod()
    f = tmp_path / "loras.txt"
    f.write_text("# the pooled best\na=x/1\n\nb=y/2   # the headline final\n")
    assert sm.load_spec({"PINQ_LORA_FILE": str(f)}) == f.read_text()
    assert sm.parse_spec(sm.load_spec({"PINQ_LORA_FILE": str(f)})) == [("a", "x/1"), ("b", "y/2")]


def test_the_environment_variable_wins_over_the_file_and_neither_is_required():
    sm = _mod()
    assert sm.load_spec({"PINQ_LORA": "a=x", "PINQ_LORA_FILE": "/nope"}) == "a=x"
    assert sm.load_spec({}) == ""


# --------------------------------------------------------------------------- resolution


def test_relative_paths_resolve_against_ART_and_absolute_ones_are_kept(tmp_path):
    sm = _mod()
    art = tmp_path / "artifacts"
    _adapter(art, "rung1-8b-headline")
    elsewhere = _adapter(tmp_path / "proj", "rung2")
    got = sm.resolve([("a", "rung1-8b-headline"), ("b", str(elsewhere))], art)
    assert got == [("a", art / "rung1-8b-headline"), ("b", elsewhere)]


def test_an_adapter_without_weights_is_refused_not_skipped(tmp_path):
    """THE BEHAVIOUR CHANGE. The old script logged `SKIPPED` and served the base alone, so a
    typo in a checkpoint path cost a queue wait and a 404 mid-sweep instead of an immediate
    failure at launch."""
    sm = _mod()
    art = tmp_path / "artifacts"
    _adapter(art, "good")
    (art / "empty").mkdir()
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        sm.check(sm.resolve([("a", "good"), ("b", "empty")], art), served_base="qwen3-8b-base")


def test_an_adapter_without_a_config_is_refused(tmp_path):
    """PEFT needs `adapter_config.json` to know the rank and the target modules; vLLM fails
    the load, but after the weights are already resident."""
    sm = _mod()
    art = tmp_path / "artifacts"
    _adapter(art, "noconf", config=False)
    with pytest.raises(ValueError, match="adapter_config.json"):
        sm.check(sm.resolve([("a", "noconf")], art), served_base="qwen3-8b-base")


def test_a_lora_name_may_not_shadow_the_served_base(tmp_path):
    """Both would be published under one model id and the base would become unreachable --
    which is the arm every trained arm is compared against."""
    sm = _mod()
    art = tmp_path / "artifacts"
    _adapter(art, "d")
    with pytest.raises(ValueError, match="qwen3-8b-base"):
        sm.check(sm.resolve([("qwen3-8b-base", "d")], art), served_base="qwen3-8b-base")


# --------------------------------------------------------------------------- the vLLM argv


def _args(sm, resolved, **kw):
    kw.setdefault("base_model", "Qwen/Qwen3-8B")
    kw.setdefault("served_base", "qwen3-8b-base")
    kw.setdefault("port", 8000)
    return sm.vllm_args(resolved=resolved, **kw)


def test_the_argv_starts_with_the_base_model_and_publishes_the_served_base(tmp_path):
    sm = _mod()
    argv = _args(sm, [])
    assert argv[0] == "Qwen/Qwen3-8B"
    assert argv[argv.index("--served-model-name") + 1] == "qwen3-8b-base"


def test_max_loras_is_the_size_of_the_map_and_never_zero(tmp_path):
    """vLLM refuses to hold more adapters at once than `--max-loras`; a fixed 4 meant the
    fifth checkpoint was a silent eviction. Zero is not a legal value, so a base-only serve
    still asks for one slot -- which is also what lets an adapter be added without a restart."""
    sm = _mod()
    for n in (0, 1, 4, 7):
        argv = _args(sm, [(f"a{i}", Path(f"/p/{i}")) for i in range(n)])
        assert argv[argv.index("--max-loras") + 1] == str(max(1, n))


def test_the_rank_is_32_because_that_is_what_every_adapter_is_trained_at():
    sm = _mod()
    argv = _args(sm, [])
    assert argv[argv.index("--max-lora-rank") + 1] == "32"


def test_max_model_len_defaults_to_16384_and_is_overridable():
    """16,384 is measured, not chosen: the 8B at bf16 is 16.31 GiB of weights and leaves
    54.46 GiB of KV cache on the 80 GB card. 8192 was the plan's untested transcription."""
    sm = _mod()
    assert sm.DEFAULT_MAX_MODEL_LEN == 16384
    argv = _args(sm, [])
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    argv = _args(sm, [], max_model_len=32768)
    assert argv[argv.index("--max-model-len") + 1] == "32768"


def test_lora_modules_is_passed_once_with_every_pair():
    """`--lora-modules a=p b=q`, one flag, n values. Repeating the flag drops all but the
    last on some vLLM releases."""
    sm = _mod()
    argv = _args(sm, [("a", Path("/p/a")), ("b", Path("/p/b"))])
    assert argv.count("--lora-modules") == 1
    i = argv.index("--lora-modules")
    assert argv[i + 1 : i + 3] == ["a=/p/a", "b=/p/b"]


def test_no_adapters_means_no_lora_modules_flag_but_lora_stays_enabled():
    sm = _mod()
    argv = _args(sm, [])
    assert "--lora-modules" not in argv
    assert "--enable-lora" in argv


def test_the_port_reaches_the_argv():
    sm = _mod()
    argv = _args(sm, [], port=8001)
    assert argv[argv.index("--port") + 1] == "8001"


# --------------------------------------------------------------------------- the record


def test_the_served_record_names_every_adapter_with_its_sha_and_the_endpoint(tmp_path):
    """`conf/checkpoints.json` says what SHOULD be served. This says what WAS -- it is the
    only artifact that can settle, after the fact, which weights answered a sweep."""
    sm = _mod()
    art = tmp_path / "artifacts"
    d = _adapter(art, "rung1-8b-headline", weights=b"headline-weights")
    rec = sm.served_record(
        resolved=[("qwen3-8b-sft-headline", d)],
        base_model="Qwen/Qwen3-8B",
        served_base="qwen3-8b-base",
        host="gpu-n629",
        port=8000,
        job="775651",
        max_model_len=16384,
    )
    assert rec["host"] == "gpu-n629"
    assert rec["port"] == 8000
    assert rec["base_model"] == "Qwen/Qwen3-8B"
    assert rec["served_base"] == "qwen3-8b-base"
    assert rec["job"] == "775651"
    assert rec["max_model_len"] == 16384
    assert rec["max_loras"] == 1
    entry = rec["adapters"]["qwen3-8b-sft-headline"]
    assert entry["path"] == str(d)
    assert entry["sha256"] == hashlib.sha256(b"headline-weights").hexdigest()
    assert entry["bytes"] == len(b"headline-weights")
    json.dumps(rec)  # must be serialisable, it is written as JSON


def test_the_record_sha_matches_what_register_checkpoint_would_compute(tmp_path):
    """Both hash `adapter_model.safetensors` and nothing else, so a served record and a
    registry row can be compared by eye."""
    sm = _mod()
    art = tmp_path / "artifacts"
    d = _adapter(art, "a", weights=b"\x00\x01\x02" * 1000)
    got = sm.sha256_file(d / "adapter_model.safetensors")
    assert got == hashlib.sha256(b"\x00\x01\x02" * 1000).hexdigest()


def test_the_base_is_in_the_served_list_even_with_no_adapters(tmp_path):
    sm = _mod()
    rec = sm.served_record(
        resolved=[],
        base_model="Qwen/Qwen3-8B",
        served_base="qwen3-8b-base",
        host="h",
        port=8000,
        job="none",
        max_model_len=16384,
    )
    assert rec["served"] == ["qwen3-8b-base"]
    assert rec["adapters"] == {}


# --------------------------------------------------------------------------- the CLI


def _run(*argv, cwd=None):
    return subprocess.run(["python3", str(SCRIPT), *argv], capture_output=True, text=True, cwd=cwd)


def test_cli_args_prints_one_vllm_argument_per_line(tmp_path):
    """serve.sh reads this with `mapfile`, so a path with a space must still be one line and
    the shell must never re-split it."""
    art = tmp_path / "artifacts"
    _adapter(art, "rung1-8b-headline")
    r = _run("args", "--spec", "qwen3-8b-sft-headline=rung1-8b-headline", "--art", str(art))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "Qwen/Qwen3-8B"
    assert f"qwen3-8b-sft-headline={art / 'rung1-8b-headline'}" in lines


def test_cli_args_exits_nonzero_when_an_adapter_is_missing(tmp_path):
    r = _run("args", "--spec", "a=nope", "--art", str(tmp_path))
    assert r.returncode != 0
    assert "adapter_model.safetensors" in r.stderr


def test_cli_record_writes_the_served_json(tmp_path):
    art = tmp_path / "artifacts"
    _adapter(art, "d", weights=b"zz")
    out = tmp_path / "SERVED.775651.json"
    r = _run(
        "record",
        "--spec",
        "a=d",
        "--art",
        str(art),
        "--host",
        "gpu-n629",
        "--job",
        "775651",
        "--out",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text())
    assert rec["adapters"]["a"]["sha256"] == hashlib.sha256(b"zz").hexdigest()
    assert rec["served"] == ["qwen3-8b-base", "a"]


# --------------------------------------------------------------------------- the shell


def test_serve_sh_is_syntactically_valid():
    r = subprocess.run(["bash", "-n", str(SERVE_SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_serve_sh_carries_no_hand_maintained_adapter_map():
    """The thing this change exists to delete. A committed map means a committed edit per
    checkpoint, and `hpc_banner` refuses to start on a dirty tree."""
    text = SERVE_SH.read_text()
    # Comments are stripped: the header EXPLAINS the arrays it deleted, and that note is worth
    # more than the two strings it contains.
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "DIRS=(" not in code
    assert "NAMES=(" not in code
    assert "PINQ_LORA" in code
    assert "serve_map.py" in code


def test_serve_sh_writes_the_served_record_before_it_execs():
    """`exec` replaces the shell; anything after it never runs."""
    text = SERVE_SH.read_text()
    assert "SERVED." in text, "the served record is named SERVED.<jobid>.json"
    assert text.index("SERVED.") < text.index("exec ")
    assert text.index("record") < text.index("exec ")


def test_serve_sh_defaults_are_the_8b_on_8000():
    text = SERVE_SH.read_text()
    assert "PINQ_BASE_MODEL:-Qwen/Qwen3-8B" in text
    assert "PINQ_SERVED_BASE:-qwen3-8b-base" in text
    assert "PINQ_MAX_MODEL_LEN:-16384" in text


def test_a_tilde_path_is_expanded_not_pasted_under_ART(tmp_path, monkeypatch):
    """`PINQ_LORA="n=~/pinq-serve/adapters/x"` is how a person writes the frozen-copy path, and
    bash does NOT expand `~` inside a quoted value. Left alone it is neither absolute nor a
    real relative path, so it would resolve to `$ART/~/pinq-serve/...` and refuse with a path
    nobody typed."""
    sm = _mod()
    monkeypatch.setenv("HOME", str(tmp_path))
    got = sm.resolve([("a", "~/pinq-serve/adapters/x")], tmp_path / "artifacts")
    assert got == [("a", tmp_path / "pinq-serve" / "adapters" / "x")]
