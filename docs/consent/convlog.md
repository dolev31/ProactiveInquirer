# Consent record: `convlog`

Subject: the Claude Code session transcripts ingested as the `convlog` suite, from two
sources:

- the Google Drive folder named "ClaudeSessions" (41 files, 25 usable conversations);
- the owner's own local session store at `~/.claude/projects/` (12 files, 10 usable).

The session that is being written while the suite is built is excluded from the second source.
A live file is not a transcript of a finished conversation, and this one contains a credential
the owner pasted into the chat.

The owner of those sessions is the author of this repository. On 2026-09-02 they stated, in
the working session that built this suite, that the transcripts are their own and that this
project has consent to ingest, scrub, label and use them as training and evaluation data.

Scope of that consent, as exercised by this code:

- the transcripts are scrubbed on ingest by `pi_eval.build.scrub`, which fails closed;
- neither the raw transcripts nor the scrubbed corpus is committed or redistributed
  (decision D16, and both paths are in `.gitignore`);
- what may be published is derived labels keyed by `item_id`, carrying no transcript text.

Third parties appear in these transcripts, because a working session quotes colleagues,
reviewers and public issues. The owner's consent covers their own sessions and cannot cover
those people, which is the second reason nothing here is redistributed and the reason the
scrubber's residual gaps (decision D17) are acceptable only under that restriction.

This file exists so the consent is a thing the build can check rather than a thing somebody
remembers. Its sha256 is recorded in `docs/DATA.md` and pinned by
`tests/test_convlog_consent_gate.py`; editing this file changes the digest and stops the
build, which is the intended behaviour.
