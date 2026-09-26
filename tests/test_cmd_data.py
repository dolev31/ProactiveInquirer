# ------------------------------------------------- `pi data status` verifies what it reports


def test_status_recomputes_the_corpus_directory_content_hash(tmp_path):
    """The corpus directory NAME is sha256(payload)[:16] and corpus_hash is inside
    semantic_hash, so bytes that no longer match the name silently change what every run_id
    built from that corpus means. Nothing recomputed it."""
    from pi_run.cmd_data import corpus_hash_of, inventory

    d = tmp_path / "data" / "corpora" / "synth" / "deadbeefdeadbeef"
    d.mkdir(parents=True)
    (d / "tasks.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')

    inv = inventory(tmp_path, "synth")
    assert inv.corpus_mismatched == ("deadbeefdeadbeef",)
    assert not inv.ok

    # and a directory that IS named for its contents passes
    real = corpus_hash_of(d / "tasks.jsonl")
    good = tmp_path / "data" / "corpora" / "synth" / real
    good.mkdir()
    (good / "tasks.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
    assert real not in inventory(tmp_path, "synth").corpus_mismatched


def test_raw_verified_counted_sidecars_and_now_counts_digests(tmp_path):
    """`sum(1 for p in files if p.with_name(p.name + ".sha256").exists())` -- present,
    therefore "verified". A file truncated mid-download reported as verified for as long as its
    sidecar survived, which is always, because nothing deletes it."""
    from pi_run.cmd_data import inventory

    raw = tmp_path / "data" / "raw" / "synth"
    raw.mkdir(parents=True)
    (raw / "f.bin").write_text("hello")
    (raw / "f.bin.sha256").write_text("0" * 64 + "\n")  # a digest that does NOT match

    unchecked = inventory(tmp_path, "synth")
    assert unchecked.raw_with_digest == 1, "the sidecar is present"
    assert unchecked.raw_verified == 0, "and nothing was checked, so nothing is verified"
    assert unchecked.ok, "not checking is not the same as failing"

    checked = inventory(tmp_path, "synth", verify_raw=True)
    assert checked.raw_verified == 0 and checked.raw_mismatched == ("f.bin",)
    assert not checked.ok


def test_a_matching_digest_verifies(tmp_path):
    import hashlib

    from pi_run.cmd_data import inventory

    raw = tmp_path / "data" / "raw" / "synth"
    raw.mkdir(parents=True)
    (raw / "f.bin").write_bytes(b"hello")
    (raw / "f.bin.sha256").write_text(hashlib.sha256(b"hello").hexdigest() + "\n")
    inv = inventory(tmp_path, "synth", verify_raw=True)
    assert inv.raw_verified == 1 and inv.ok


def test_data_build_passes_raw_dir_through():
    """`--raw-dir` was registered, documented in --help, accepted by build_suite -- and never
    passed. A user pointing at a different pinned-input directory got a build from the default
    one, with no error and a corpus_hash printed as though it came from their inputs."""
    import inspect

    from pi_run import cmd_data

    src = inspect.getsource(cmd_data.cmd_data_build)
    assert "raw_dir=" in src, "the flag must reach build_suite"
    assert "raw_dir" in inspect.signature(cmd_data.build_suite).parameters
