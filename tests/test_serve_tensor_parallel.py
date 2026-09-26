"""A multi-GPU serve must be expressible, and asking for one must not change the 8B's argv.

WHY THIS EXISTS. `scripts/hpc/serve.sh` could only ever submit `-gpu "num=1"`: there was no
spelling anywhere under `scripts/hpc/` for `--tensor-parallel-size` (`grep -rn
'tensor-parallel\\|tensor_parallel' scripts/hpc/` returned nothing on 2026-09-18), so
Qwen3-32B -- 61.02 GiB of bf16 weights, measured by summing the 17 safetensors shards in the
cluster's own HF snapshot -- had no configuration on one 80 GiB card that leaves room for the
HARD-CODED `--max-model-len 16384`, a number that was measured FOR THE 8B (2.25 GiB of KV for
one full-length sequence, against 4.00 GiB for the 32B).

THE DISCIPLINE THIS FOLLOWS, and the reason for the first test below. `SHA_OMIT_WHEN_DEFAULT`
(`src/pinq_train/rung1_sft/train.py` ~174) exists because "the knob was set to its default" and
"the knob did not exist" are two different facts that must not produce the same bytes -- or,
read the other way round, because introducing a knob must not move the identity of every run
that predates it. The same rule applies to an argv: `PINQ_TENSOR_PARALLEL` UNSET must produce
the argv the 8B has always been served with, argument for argument, with no
`--tensor-parallel-size` and above all no empty string for `argparse` to have to parse. An
explicit `1` is NOT the same thing as unset: it is a stated choice about topology and it is
recorded as one.

`artifacts/rung32b_serve_20260918/argv_before.txt` is that argv, captured from the unmodified
`serve_map.py` at HEAD 9884045 before a line of this was written; `GOLDEN_8B_ARGV` below is a
transcription of it and the byte-identity test compares against it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hpc" / "serve_map.py"
SERVE_SH = REPO / "scripts" / "hpc" / "serve.sh"

# Captured from `python3 scripts/hpc/serve_map.py args --spec ... --art <tmp>` at commit
# 9884045, BEFORE the tensor-parallel change existed. The adapter path is substituted per-run;
# everything else is frozen.
GOLDEN_8B_ARGV = [
    "Qwen/Qwen3-8B",
    "--served-model-name",
    "qwen3-8b-base",
    "--enable-lora",
    "--max-lora-rank",
    "32",
    "--max-loras",
    "1",
    "--max-model-len",
    "16384",
    "--port",
    "8000",
    "--lora-modules",
]


def _mod():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist")
    spec = importlib.util.spec_from_file_location("serve_map_tp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _adapter(root: Path, name: str, *, weights: bytes = b"w") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(weights)
    (d / "adapter_config.json").write_text(json.dumps({"r": 32}))
    return d


def _args(sm, resolved, **kw):
    kw.setdefault("base_model", "Qwen/Qwen3-8B")
    kw.setdefault("served_base", "qwen3-8b-base")
    kw.setdefault("port", 8000)
    return sm.vllm_args(resolved=resolved, **kw)


# ------------------------------------------------------------------ unset is not a value


def test_unset_tensor_parallel_is_the_argv_the_8b_has_always_had():
    """Byte-identical, argument for argument, against the argv captured at HEAD 9884045.

    A new knob that shifts the argv of every serve that does not use it is a knob that
    invalidates every `ARGS.<job>.txt` already on disk. Passing `tensor_parallel=None`
    explicitly is the point of the test: it proves the parameter EXISTS and that its absent
    state emits nothing, which is a stronger statement than calling `vllm_args` without it.
    """
    sm = _mod()
    d = Path("/p/a")
    argv = _args(sm, [("qwen3-8b-sft-headline", d)], tensor_parallel=None)
    assert argv == [*GOLDEN_8B_ARGV, f"qwen3-8b-sft-headline={d}"]
    assert "--tensor-parallel-size" not in argv
    assert "" not in argv, "an empty argument is one argparse has to parse, and vLLM refuses it"


def test_the_default_call_and_the_explicitly_unset_call_agree():
    """Omitting the parameter and passing None must not be two behaviours."""
    sm = _mod()
    assert _args(sm, []) == _args(sm, [], tensor_parallel=None)


# ------------------------------------------------------------------ set emits the flag


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_a_set_tensor_parallel_emits_the_flag_once_with_its_value(n):
    """Including n=1: an explicit 1 is a STATED topology, not the absence of one. It is the
    same rule as `SHA_OMIT_WHEN_DEFAULT` read forwards -- unset is dropped, any value that was
    actually chosen is carried, even the one that happens to match vLLM's own default."""
    sm = _mod()
    argv = _args(sm, [], tensor_parallel=n)
    assert argv.count("--tensor-parallel-size") == 1
    assert argv[argv.index("--tensor-parallel-size") + 1] == str(n)


def test_the_flag_is_added_and_nothing_else_moves():
    """The rest of the argv must be the golden one, in the same order."""
    sm = _mod()
    argv = _args(sm, [], tensor_parallel=2)
    i = argv.index("--tensor-parallel-size")
    without = argv[:i] + argv[i + 2 :]
    assert without == GOLDEN_8B_ARGV[:-1]  # no adapters -> no --lora-modules


# ------------------------------------------------------------------ the environment


def test_the_env_parser_reads_unset_and_empty_as_absent():
    """`PINQ_TENSOR_PARALLEL=""` is what an `export` with nothing after it leaves behind, and
    what `${PINQ_TENSOR_PARALLEL:-}` expands to under `set -u`. Both must mean the 8B's argv,
    NOT `--tensor-parallel-size ''`."""
    sm = _mod()
    assert sm.parse_tensor_parallel(None) is None
    assert sm.parse_tensor_parallel("") is None
    assert sm.parse_tensor_parallel("   ") is None


def test_the_env_parser_reads_a_number():
    sm = _mod()
    assert sm.parse_tensor_parallel("2") == 2
    assert sm.parse_tensor_parallel(" 2 ") == 2
    assert sm.parse_tensor_parallel(2) == 2


@pytest.mark.parametrize("bad", ["0", "-1", "two", "2.0", "2,2"])
def test_the_env_parser_refuses_anything_that_is_not_a_gpu_count(bad):
    """A typo here is a job that holds N GPUs for its whole wall limit and serves nothing.
    `int("2.0")` already raises; `0` and `-1` parse fine and must be refused by name."""
    sm = _mod()
    with pytest.raises(ValueError):
        sm.parse_tensor_parallel(bad)


# ------------------------------------------------------------------ the CLI


def _run(*argv, env=None):
    e = dict(os.environ)
    e.pop("PINQ_TENSOR_PARALLEL", None)
    e.update(env or {})
    return subprocess.run(["python3", str(SCRIPT), *argv], capture_output=True, text=True, env=e)


def test_cli_with_the_env_unset_prints_the_golden_argv(tmp_path):
    art = tmp_path / "artifacts"
    d = _adapter(art, "rung1-8b-headline")
    r = _run("args", "--spec", "a=rung1-8b-headline", "--art", str(art))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [*GOLDEN_8B_ARGV, f"a={d}"]


def test_cli_with_the_env_EMPTY_prints_the_golden_argv(tmp_path):
    """The failure this forbids: `--tensor-parallel-size` followed by a blank line, which
    `serve.sh` reads as an argument and vLLM rejects after the queue wait is spent."""
    art = tmp_path / "artifacts"
    d = _adapter(art, "rung1-8b-headline")
    r = _run(
        "args", "--spec", "a=rung1-8b-headline", "--art", str(art), env={"PINQ_TENSOR_PARALLEL": ""}
    )
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines == [*GOLDEN_8B_ARGV, f"a={d}"]
    assert "" not in lines


def test_cli_reads_PINQ_TENSOR_PARALLEL_from_the_environment(tmp_path):
    art = tmp_path / "artifacts"
    _adapter(art, "rung1-8b-headline")
    r = _run(
        "args",
        "--spec",
        "a=rung1-8b-headline",
        "--art",
        str(art),
        env={"PINQ_TENSOR_PARALLEL": "2"},
    )
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[lines.index("--tensor-parallel-size") + 1] == "2"


def test_cli_refuses_a_bad_tensor_parallel_before_it_costs_a_gpu_hour(tmp_path):
    art = tmp_path / "artifacts"
    _adapter(art, "rung1-8b-headline")
    r = _run(
        "args",
        "--spec",
        "a=rung1-8b-headline",
        "--art",
        str(art),
        env={"PINQ_TENSOR_PARALLEL": "0"},
    )
    assert r.returncode != 0
    assert "PINQ_TENSOR_PARALLEL" in r.stderr


def test_the_explicit_flag_wins_over_the_environment(tmp_path):
    art = tmp_path / "artifacts"
    _adapter(art, "rung1-8b-headline")
    r = _run(
        "args",
        "--spec",
        "a=rung1-8b-headline",
        "--art",
        str(art),
        "--tensor-parallel",
        "4",
        env={"PINQ_TENSOR_PARALLEL": "2"},
    )
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[lines.index("--tensor-parallel-size") + 1] == "4"


# ------------------------------------------------------------------ the record


def test_the_served_record_says_which_topology_answered(tmp_path):
    """`SERVED.<job>.json` is the only artifact that can settle after the fact what was
    running. A 32B answered by two cards and a 32B answered by one are different machines with
    different KV budgets, so the number belongs in the record; unset stays `null` rather than
    becoming a `1` nobody chose."""
    sm = _mod()
    art = tmp_path / "artifacts"
    d = _adapter(art, "qwen3-32b-sft-headline", weights=b"32b")
    rec = sm.served_record(
        resolved=[("qwen3-32b-sft-headline", d)],
        base_model="Qwen/Qwen3-32B",
        served_base="qwen3-32b-base",
        host="gpu-n624",
        port=8003,
        job="0",
        tensor_parallel=2,
    )
    assert rec["tensor_parallel_size"] == 2
    json.dumps(rec)
    bare = sm.served_record(
        resolved=[],
        base_model="Qwen/Qwen3-8B",
        served_base="qwen3-8b-base",
        host="h",
        port=8000,
        job="0",
    )
    assert bare["tensor_parallel_size"] is None


# ------------------------------------------------------------------ the shell


def test_serve_sh_is_still_syntactically_valid():
    r = subprocess.run(["bash", "-n", str(SERVE_SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_serve_sh_passes_the_flag_only_when_the_env_is_non_empty():
    """The shell half of "unset is not a value". An unconditional
    `--tensor-parallel "${PINQ_TENSOR_PARALLEL:-}"` would hand serve_map an empty string on
    every 8B serve -- which is exactly the empty argument the CLI tests above forbid."""
    code = "\n".join(
        ln for ln in SERVE_SH.read_text().splitlines() if not ln.lstrip().startswith("#")
    )
    assert "PINQ_TENSOR_PARALLEL" in code
    assert "--tensor-parallel" in code
    # the append must be guarded by a non-empty test, not pasted into MAPARGS unconditionally
    assert 'if [ -n "${TENSOR_PARALLEL:-}" ]' in code or 'if [ -n "$TENSOR_PARALLEL" ]' in code


def test_serve_sh_defaults_are_unchanged():
    """The 8B's three defaults are load-bearing elsewhere (tests/test_serve_map.py pins them
    too); a multi-GPU knob must not have moved any of them."""
    text = SERVE_SH.read_text()
    assert "PINQ_BASE_MODEL:-Qwen/Qwen3-8B" in text
    assert "PINQ_SERVED_BASE:-qwen3-8b-base" in text
    assert "PINQ_MAX_MODEL_LEN:-16384" in text
