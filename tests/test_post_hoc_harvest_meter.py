"""The post-simulate harvest must not delete a run the campaign completed.

WHAT THIS REPLAYS, AND WHY IT IS A FIXTURE AND NOT A SCAN. `_attach_env_evidence` charges a
finished dialogue's transcript tool calls through `meter_env_calls`, against the SAME hard cap
`run_loop` already charged the in-loop Asks to. When the sum exceeds the cap the charge raises
`BudgetExceeded`, the runner's own `except Exception` catches it, and a unit that ran to
completion and produced a real reward is written out as `status: error`. Nothing is prevented:
the Orchestrator executed those calls before this function ever ran. A completed measurement is
deleted instead.

THE POPULATION IS THE PUBLISHED ONE. The 204 rows below are the tau2_retail/test fork runs that
`artifacts/forks_test/forks.tau2_retail.test.json` reports -- the two questioner arms of the
transfer result -- selected by exactly the predicate `scripts/report_forks.py` uses. `runs/` is
gitignored and absent from CI, so the scan is pinned here rather than re-run. The scan is a
COMMITTED tool, not a scratch script, so every number below can be re-derived:

    python scripts/replay_post_hoc_harvest.py --suite tau2_retail --split test
    -> incomplete run dirs skipped (reported, not dropped): 2
           151cdb3eb7145cc253fcdbb865372525 (missing ['outcome.json'])
           8025457b01a7eb1f506ba8a691c9b6dd (missing ['outcome.json'])
       population: 204 runs
       run_ids_sha: 1a060380806fff3e
       code_versions: {'922060c': 68, '828a720': 136}
       statuses: {'ok': 204}
       caps: {16: 204}
       ledger ordering ok on every row: True
       instrument disagreement: 0 of 204
       the pre-change harvest refuses: 70 of 204
         by arm: {'inquirer_prompted': 13, 'self_ask': 57}
         inquirer_prompted    mean charged =  9.941  (in-loop  8.637 + harvest  1.304)  cap 16
         self_ask             mean charged = 18.304  (in-loop 10.843 + harvest  7.461)  cap 16
       pairs 102 -> 41 ; dialogue clusters 32 -> 19 ; fork points 34 -> 21
       as published                   pairs=102 diff_mean=-4.3137  fewer/more=71/24
       surviving a pre-change re-run  pairs= 41 diff_mean=-0.2927  fewer/more=18/17

The two skipped directories are another session's live sweep, not part of the published
population; the run-id digest above is unchanged by them, which is how that is known. `runs/` is
shared and live, so that digest -- not the suite/split predicate -- is what pins this population.
`--emit-fixture` prints the table below, so it is regenerated rather than hand-edited.

TWO INDEPENDENT INSTRUMENTS AGREED ON EVERY ROW, which is why `n_post` is trustworthy.
`ledger.jsonl`'s `retrieval_calls` rows split cleanly by `hard`: the in-loop charges come from
`charge_retrieval` (hard, cap 16) and the harvest's from `ledger.record` (not hard, no cap),
whose only call site at both campaign code versions is `meter_env_calls` itself (`git grep -n
'record("retrieval_calls"' 828a720 -- src` returns one line). Independently, `outcome.json`'s
`env_calls` gives the recorded transcript, and its count of `ok=True` calls -- which is what
`meter_env_calls` charges -- equals the ledger's harvest count on all 204 rows. The ordering
assumption was checked rather than assumed: no in-loop charge follows a harvest charge on any
row.

`run_id`, `arm_id`, `seed`, `foreign_prefix_k`, `budget_cap`, the in-loop spend and the harvest
call count are the real measured values. Only `foreign_trace_sha` is relabelled, to T0..T31 for
width; it is read here only to count dialogue clusters, and the relabelling is injective by
construction.

WHAT THE PRE-CHANGE CODE DOES TO THIS POPULATION, measured, and why the asymmetry is the
finding rather than the count. It refuses 70 of the 204 -- 57 of the 102 control runs against 13
of the 102 treatment runs -- because `self_ask` averages 18.3 charged retrieval calls against a
cap of 16 while `inquirer_prompted` averages 9.9. The deletion therefore selects on the endpoint:
the control arm's longest dialogues are the expensive ones. 102 pairs collapse to 41 and 32
recorded dialogues to 19, and on the 41 pairs that survive, the published effect of
$-4.3137$ follow-up turns reads $-0.2927$ with a sign split of 18/17 instead of 71/24. A
quantity that is filtered by its own outcome is not a measurement of the thing it names.

Inside `run_loop` the cap is a decision the policy is still making, and exceeding it stops the
rollout with `stop_reason="budget"` -- that is the hard cap doing its job and it must keep doing
it (`test_the_in_loop_cap_still_refuses`, below). Here the dialogue is over and the arithmetic
is after the fact, so the spend is RECORDED and the overrun REPORTED rather than thrown.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from pinq.budget import BudgetExceeded, BudgetLedger

# The published population's digest, recomputed from the run ids below so the fixture cannot
# drift from the artifact it claims to replay without this test saying so.
RUN_IDS_SHA = "1a060380806fff3e"

# columns: run_id, arm_id, seed, foreign_trace_sha (relabelled), foreign_prefix_k,
#          in-loop retrieval_calls at harvest time, harvest tool calls charged
_ROWS: list[tuple[str, str, int, str, int, int, int]] = [
    ("00b15fb4755e555794b2dee758fbc598", "inquirer_prompted", 0, "T0", 16, 4, 0),
    ("01ce9ca8861e6263c8af6b8eb7709423", "inquirer_prompted", 2, "T1", 2, 16, 2),
    ("01d123ce8e139d7a3c4e39c8825de707", "inquirer_prompted", 0, "T2", 2, 5, 0),
    ("02d67b2d4178b76e0bae0439a2650e6b", "self_ask", 0, "T3", 14, 1, 1),
    ("030b8083a51038a90ed6117cd21e3dde", "inquirer_prompted", 2, "T4", 6, 6, 1),
    ("0408ac8a58dfc2d16216ed61d5c7fbfb", "inquirer_prompted", 1, "T5", 16, 12, 2),
    ("04b22c1052d6cdfac5be9fbc8a277db7", "inquirer_prompted", 0, "T6", 34, 10, 3),
    ("0611bef6b06fe593d01c3b48eabfffdb", "self_ask", 2, "T7", 2, 2, 2),
    ("065bc8ed3cc3c297c1e34e818f9a652d", "inquirer_prompted", 2, "T8", 8, 6, 0),
    ("09c37e298792cce413f28c6668bfed52", "self_ask", 1, "T5", 16, 7, 2),
    ("0b4e603e1871c4cab6b1adb76cba94b7", "self_ask", 2, "T9", 4, 16, 7),
    ("0c5d577fe3ab87e8a0c8316f49bac3cd", "self_ask", 2, "T10", 21, 6, 5),
    ("0d88cf4c65aec249180b4a8698877947", "self_ask", 0, "T11", 11, 1, 6),
    ("0e83369cf2b26fe1022a28d4d32a2201", "self_ask", 1, "T0", 16, 16, 8),
    ("0ef3cc5aa5da5c99185f27adf722d4ef", "self_ask", 1, "T12", 4, 16, 10),
    ("0efd0494f1f8e4bb6cfd376c41cd4005", "inquirer_prompted", 2, "T12", 4, 6, 3),
    ("0f6791b15b721bc87e50c42ffdb28d96", "inquirer_prompted", 1, "T13", 4, 5, 1),
    ("117493a788d6126b67dfa4cd6b180d56", "inquirer_prompted", 0, "T12", 4, 5, 3),
    ("11d73c7a336bdcca8f34cb2c8c58d017", "self_ask", 0, "T14", 13, 11, 13),
    ("124a333ed972ddf88013b7344e7dfa74", "self_ask", 1, "T11", 11, 2, 1),
    ("138d077aacbdbe8cb3e691e676672bbb", "self_ask", 1, "T15", 4, 16, 15),
    ("149437e09d9cb0d3d0d07899d2ed93e2", "self_ask", 1, "T16", 21, 16, 20),
    ("14eed77b0c5c50757f5edd813bf8dc34", "inquirer_prompted", 0, "T17", 12, 3, 1),
    ("1520dcbe5ae68b4d127510343755ed9f", "self_ask", 2, "T5", 16, 11, 4),
    ("1593c13e99c95c9d3748103768da94d9", "self_ask", 2, "T18", 11, 16, 12),
    ("15e514085f9ecfc43c3fbdbd19914261", "inquirer_prompted", 1, "T19", 4, 15, 0),
    ("19da72693b3d3bb6cf5c876f70b13ab9", "self_ask", 2, "T4", 26, 3, 1),
    ("1ac9978754279fc319b33cae9610dca0", "self_ask", 2, "T13", 4, 2, 6),
    ("1af6e171cea66836f6107e63cc2a85a0", "self_ask", 2, "T20", 6, 11, 9),
    ("1b8e61c832ad74058b7990ad599b6df3", "self_ask", 2, "T12", 4, 5, 1),
    ("1cbafdd32f261996fa2172d1e59e54cc", "inquirer_prompted", 0, "T21", 14, 9, 0),
    ("1d8ba18ff8c750069365632e9b751936", "inquirer_prompted", 2, "T9", 4, 7, 1),
    ("1e2940d7e300374b5b7eb30db0a506b3", "inquirer_prompted", 0, "T18", 11, 3, 1),
    ("1f2eb342aa5a2c8efe4b92e29b4a256b", "inquirer_prompted", 2, "T20", 6, 8, 1),
    ("2030530e24102c952d12a91b7fba3a3d", "inquirer_prompted", 1, "T14", 13, 16, 2),
    ("2050578cdd5f972fc5e6bea362a37538", "inquirer_prompted", 1, "T16", 21, 16, 0),
    ("2265ace56404c37e8924a5c4a12ef242", "self_ask", 1, "T18", 11, 16, 6),
    ("2439b22c678a12be91cd4df615d56e5c", "inquirer_prompted", 1, "T21", 14, 8, 0),
    ("24503747bf83d80bb1ff81b4883e8c46", "self_ask", 0, "T8", 8, 14, 5),
    ("25793e75293266a1a2e96b9972c71351", "self_ask", 2, "T1", 2, 10, 5),
    ("25b71a049f60a25f3690663835665336", "inquirer_prompted", 1, "T6", 34, 11, 4),
    ("25dbf095d939c52d684c571d94117105", "inquirer_prompted", 2, "T10", 21, 13, 2),
    ("25f7af2dd5b908e8e276a7bfb7065ec7", "self_ask", 0, "T6", 34, 14, 6),
    ("297097c02b9fc5cdd554ef2a96fb9a01", "self_ask", 0, "T4", 26, 0, 2),
    ("29cdef708ab0b8c6f5a129018d22e197", "self_ask", 1, "T13", 4, 16, 7),
    ("2c7a211901c5917b6591c97e353cda97", "inquirer_prompted", 0, "T22", 20, 16, 4),
    ("2d650137a90ac380a968575ce6492acf", "self_ask", 2, "T23", 4, 4, 5),
    ("2d73a66301e709fa915a8f97ab268e5d", "self_ask", 2, "T3", 14, 1, 1),
    ("2f4cd8a24244d8a5e9bd22918a350b69", "inquirer_prompted", 0, "T23", 4, 7, 1),
    ("34fe8910ee5f02dbf6cfc71d71182897", "self_ask", 2, "T16", 21, 11, 6),
    ("35a553c28f749e675ae88a2c60248b6b", "self_ask", 1, "T1", 2, 16, 11),
    ("371eba952a9a30bee1641b9b182e7969", "inquirer_prompted", 1, "T24", 25, 6, 1),
    ("38b71dc3be405e49fe7cffce53c9ef96", "self_ask", 0, "T10", 21, 5, 5),
    ("39be788d74b28205f967e77fdf0becae", "self_ask", 1, "T12", 18, 2, 1),
    ("3ca0053c6961ad28d54cf3d510622029", "inquirer_prompted", 1, "T8", 8, 5, 0),
    ("3e1f8673a27b218fedb040a181d11f3c", "self_ask", 2, "T25", 21, 4, 3),
    ("40f39382231756a527e8abcd78eaeacc", "self_ask", 1, "T26", 28, 0, 1),
    ("4282907e06a8b6f928b0f450b513f3ab", "inquirer_prompted", 1, "T2", 2, 9, 1),
    ("45894caf7853eab5dc3da35ab0c09edf", "self_ask", 0, "T12", 18, 6, 2),
    ("4734988c3b31a9ced52733081e02a6f7", "self_ask", 2, "T6", 34, 13, 5),
    ("48cd8dfc5e743380bb212326c2ad47f5", "self_ask", 0, "T26", 28, 4, 4),
    ("49920438fc7a620789a88cb595ced542", "self_ask", 1, "T9", 4, 9, 5),
    ("4a1af1958b2e6460e98c851f35c54ee8", "self_ask", 2, "T27", 15, 16, 25),
    ("4bda501eb546a3817c27385dce804cc4", "inquirer_prompted", 0, "T9", 4, 7, 1),
    ("504aaab4c6a6257dae11cf62c6fab6c6", "inquirer_prompted", 0, "T28", 2, 6, 0),
    ("5165af3a0fecd6b7429dc798fe078d48", "inquirer_prompted", 0, "T19", 4, 16, 3),
    ("52b2be76bc20571d4fec2ff6f98ee5a2", "inquirer_prompted", 2, "T11", 11, 6, 1),
    ("52dcc848fe6820f54e5918a0411a986d", "inquirer_prompted", 2, "T25", 21, 6, 2),
    ("52e041f378d4a2b7269e6be1edf26ee0", "inquirer_prompted", 2, "T24", 25, 9, 0),
    ("56e7bb975ef9f23aa47acf662975a4d0", "self_ask", 0, "T25", 21, 10, 3),
    ("571a6afb97a0c596f25e2642e2a26d9a", "inquirer_prompted", 2, "T14", 13, 16, 4),
    ("5751997dce9e675e32cfcb3ae1aa7c3c", "self_ask", 1, "T24", 25, 16, 9),
    ("5845cecf14922a04ab1d3376ade9836c", "inquirer_prompted", 2, "T0", 16, 6, 0),
    ("59958d13a492b1084b0cd83288974f4f", "inquirer_prompted", 1, "T10", 21, 9, 1),
    ("5a11b1bc7d8297057ff14e7afe9bd064", "self_ask", 0, "T29", 16, 16, 23),
    ("5a3402f8cd4badc11bedcd408caad4ad", "inquirer_prompted", 1, "T12", 18, 5, 3),
    ("5b36a303bafab75c7ed3a6e6835f86f1", "inquirer_prompted", 1, "T26", 28, 3, 1),
    ("5e56ffdef0dbc901b160a5aecdeb8981", "inquirer_prompted", 2, "T22", 20, 11, 2),
    ("5f4795830e0cea528945309ee3071b2e", "inquirer_prompted", 1, "T30", 2, 10, 0),
    ("62bd81a682a8db552a4b2b1bb4992927", "self_ask", 0, "T16", 21, 16, 11),
    ("634f38a9f6647810307cfe9796754acb", "inquirer_prompted", 0, "T1", 2, 11, 1),
    ("6487b279112b40640334f8e185e30f86", "inquirer_prompted", 0, "T3", 14, 6, 0),
    ("65ba654f1ca10dba8efde2852c2c707a", "self_ask", 1, "T17", 12, 16, 9),
    ("67c58a621ab9aacec377fdcb0ecd005e", "inquirer_prompted", 2, "T27", 15, 7, 2),
    ("6897835300ce97c21583abaae06ff671", "self_ask", 2, "T2", 2, 16, 13),
    ("691c08c9d0aba60710881d536fa9f35e", "self_ask", 2, "T14", 13, 16, 29),
    ("6931767ed06808141e3d44a182cf94c3", "inquirer_prompted", 0, "T5", 16, 6, 0),
    ("69f887f4397cb5fc52e77c9fb3b557c6", "self_ask", 0, "T9", 4, 16, 8),
    ("6c72b529c2b53b48ee1cd8ecfddebda6", "self_ask", 1, "T28", 2, 16, 10),
    ("6e5132b8039418b5f9e93477921edc85", "inquirer_prompted", 0, "T11", 11, 7, 1),
    ("6f4dd4e9c1d59ebd7a72bb3fe33da863", "self_ask", 2, "T30", 2, 16, 8),
    ("6f6c4f93201173b600915b439cfa47a6", "self_ask", 0, "T2", 2, 16, 19),
    ("6fa689d457ea9f8d3787cf7a68ae6968", "self_ask", 2, "T15", 4, 8, 0),
    ("72933ac8dfe92f9d542671bd15e6c980", "self_ask", 0, "T31", 13, 16, 9),
    ("72a56f835b4866a4de2a6d56f8a10f16", "inquirer_prompted", 0, "T4", 26, 4, 0),
    ("733c6f6a3e5603b2d28be727ed243dec", "self_ask", 1, "T4", 6, 16, 7),
    ("7485410a5b8c02b61cd17ffafe9aed25", "self_ask", 2, "T12", 18, 4, 3),
    ("756c54cfc71bd3aa5ce3fe121998203c", "self_ask", 1, "T21", 14, 10, 2),
    ("7699e696e79b9447e356db738fbb5e77", "inquirer_prompted", 0, "T30", 2, 12, 1),
    ("7c0933e4a89a1734b02cc82716208d3b", "self_ask", 2, "T4", 6, 10, 1),
    ("7ce8c2ca51edafb07c9e7282da72e51c", "inquirer_prompted", 1, "T12", 4, 6, 2),
    ("7d46d193ab2df936a49361f3275d15d4", "self_ask", 1, "T6", 34, 16, 12),
    ("7e197e1fddbb08b32e4d41ad6ebe96f7", "self_ask", 2, "T11", 11, 16, 24),
    ("7eab106140e3888ae5b31a058958c6cf", "inquirer_prompted", 2, "T7", 2, 10, 1),
    ("7fddb3774462206cfc3b1c0766e1fb70", "self_ask", 2, "T22", 20, 16, 18),
    ("81f2f1761bbbaa64f223b6b97e39106f", "self_ask", 1, "T19", 4, 7, 7),
    ("84989fbbdf7b96534973907fbec7b7f5", "inquirer_prompted", 1, "T0", 16, 4, 0),
    ("858dd2f2e26a185eddcda7f6b6426ee8", "self_ask", 0, "T19", 4, 5, 7),
    ("86d8c6a74d17f15c2a533bfe7be04584", "inquirer_prompted", 1, "T18", 11, 3, 1),
    ("88f05388b0fddf43cf917d45abd7e234", "self_ask", 1, "T4", 26, 4, 2),
    ("89a87f838714d68abb12a8bee655f6f8", "inquirer_prompted", 0, "T14", 13, 14, 3),
    ("8b66e4c64ca0a1dc72cec1411206604e", "inquirer_prompted", 0, "T25", 21, 10, 1),
    ("8df9b9c30d8b9fb91d746c66b1ee18d2", "inquirer_prompted", 2, "T4", 26, 5, 1),
    ("8efa577ff7209d7909d92b992f6c7cbc", "inquirer_prompted", 2, "T18", 11, 3, 1),
    ("8f082711182056aaea065038e211c598", "inquirer_prompted", 1, "T17", 12, 3, 1),
    ("8fde84b19a97d95aa05b2d426d644ab0", "self_ask", 0, "T4", 6, 11, 4),
    ("9181e0dc5df00bbd5c75145bc2cb71a8", "inquirer_prompted", 1, "T9", 4, 6, 1),
    ("93ccd90d200378ec991ea736ce5c2cd2", "self_ask", 0, "T30", 2, 16, 1),
    ("9447ce921f85db732778cbcc64eb217e", "inquirer_prompted", 2, "T21", 14, 8, 0),
    ("945a865afc040f26d380ca3e56e1a273", "inquirer_prompted", 1, "T29", 16, 16, 0),
    ("950737cc061bbaabfea2a168c95fb008", "self_ask", 1, "T27", 15, 16, 12),
    ("95630f11be70ad02dd4247c7bd2ced61", "self_ask", 2, "T17", 12, 8, 4),
    ("95b56cc641f471dc9b803415cf786d21", "self_ask", 1, "T22", 20, 4, 3),
    ("95c5f4fc7fb899af2ad53219b5610b1f", "inquirer_prompted", 2, "T16", 21, 12, 1),
    ("9766ad8e49831470e112fd6075683084", "inquirer_prompted", 2, "T28", 2, 8, 2),
    ("98ee840beb05ae27c281ebe7af1f1f3e", "self_ask", 2, "T28", 2, 16, 5),
    ("99310b62a0d708ce6257785b6ba3fe87", "self_ask", 2, "T24", 25, 4, 2),
    ("9cef936a4d7fef46441b0d167a388f61", "inquirer_prompted", 2, "T2", 2, 8, 1),
    ("9dad6aad69a2351ddec35c2f7507cf58", "inquirer_prompted", 2, "T30", 2, 13, 3),
    ("9e3dcb0b658a707c7c563fd03b34356f", "inquirer_prompted", 1, "T25", 21, 7, 1),
    ("9fae24f7d51e7b0a7375dadaf6811505", "self_ask", 0, "T23", 4, 8, 3),
    ("a0cb149c9dad24427c6ad2879e21e223", "self_ask", 2, "T26", 28, 16, 6),
    ("a2568452b973926c2973662d5e711d91", "self_ask", 1, "T20", 6, 6, 10),
    ("a3221eacafb10a79ab0d2999552f9758", "self_ask", 1, "T8", 8, 2, 0),
    ("a3f14c2c884a4d2154ec4e0b14bc63e0", "inquirer_prompted", 1, "T22", 20, 16, 4),
    ("a7a442dd12453f92460abd25a95c3ff2", "inquirer_prompted", 1, "T15", 4, 16, 3),
    ("a7eb6eaa45eabfe0835772146f7ecbaa", "inquirer_prompted", 1, "T3", 14, 6, 0),
    ("a914720069e781e7a36b9a1d0aefc409", "inquirer_prompted", 2, "T29", 16, 13, 0),
    ("a914ad0783707ba112263f7d99fdb7c2", "inquirer_prompted", 2, "T15", 4, 16, 3),
    ("ab18aad49423e3f2bf423572a4411161", "inquirer_prompted", 0, "T12", 18, 5, 2),
    ("acf6fb9d716f21e5d90d6bb563f038d6", "inquirer_prompted", 2, "T31", 13, 11, 0),
    ("adb2ee62c0b3e488108c9dba6be222f1", "self_ask", 0, "T17", 12, 14, 18),
    ("b125e981fb8a14008dad08a305cbdd18", "self_ask", 1, "T23", 4, 12, 12),
    ("b186339ebdab940fedc1d5115063aa40", "self_ask", 0, "T20", 6, 16, 22),
    ("b29db4eb1e264b3671538fd0810f4765", "self_ask", 0, "T18", 11, 16, 12),
    ("b50fc41629dc69b8d0367fe320c94e70", "inquirer_prompted", 1, "T11", 11, 6, 2),
    ("b72b01ffd03539ac18de9a7b4d049408", "self_ask", 2, "T0", 16, 11, 6),
    ("ba1dd9774683d39da93295c92d8e8503", "inquirer_prompted", 1, "T27", 15, 10, 2),
    ("ba321645c79535f794ed1ded7e6d9e3f", "inquirer_prompted", 1, "T7", 2, 9, 0),
    ("bba0721ed29e727ac5a3891ce17dc8e7", "inquirer_prompted", 0, "T15", 4, 16, 7),
    ("bc22a7421ff6b6b02b715d41875e991e", "self_ask", 2, "T21", 14, 16, 8),
    ("bc3118b28af48f26c722e91bbbd94564", "self_ask", 1, "T25", 21, 16, 13),
    ("bf1ad2d2272c8a3b52e4a9e843140df2", "inquirer_prompted", 0, "T8", 8, 9, 0),
    ("c0a88a792a95211f04e659ec88b7a5e1", "self_ask", 1, "T7", 2, 8, 6),
    ("c47d58cf670bcf3ebab6040cbd85e205", "inquirer_prompted", 2, "T17", 12, 3, 1),
    ("c76f41b02b47897ebde3ad2f47f3532f", "inquirer_prompted", 0, "T13", 4, 5, 1),
    ("c815adee76b2cc4e2c4f3c35f8db17a8", "inquirer_prompted", 1, "T4", 6, 6, 1),
    ("c8b0911af23e47073c449850f7ba6a35", "inquirer_prompted", 1, "T23", 4, 6, 1),
    ("cc0ac75d23c8a64ea103802a41d85d1c", "self_ask", 1, "T29", 16, 2, 1),
    ("cc270a3e44310aaf27464bad3eaa5925", "inquirer_prompted", 0, "T29", 16, 10, 0),
    ("cd1bfe6924cd481b200be92137da0d79", "self_ask", 2, "T19", 4, 16, 8),
    ("cf861786b44a64a6fbb2aa82bcec372f", "inquirer_prompted", 0, "T10", 21, 13, 1),
    ("d0cd3b99fe43f0802a1b61c1c96ae8b4", "self_ask", 0, "T27", 15, 16, 5),
    ("d12a657f36af998323cd22c702080dd9", "self_ask", 2, "T31", 13, 16, 6),
    ("d13bdae7edda5fdcd3e6d4c23d63abdb", "self_ask", 0, "T22", 20, 16, 17),
    ("d1a62650f9716c8c09dcea7b88986e2b", "self_ask", 1, "T3", 14, 0, 1),
    ("d23708e41ca1d97a72a23a5621b0f94e", "self_ask", 0, "T1", 2, 13, 4),
    ("d297ec7db0c3cd257055dee88185b870", "inquirer_prompted", 0, "T16", 21, 14, 2),
    ("d35b39a955aab42d468cc8b0761f2d68", "inquirer_prompted", 2, "T23", 4, 7, 1),
    ("d5cebd553d3cfdcbbe9003c761c068e9", "inquirer_prompted", 2, "T5", 16, 10, 1),
    ("d5d55852cb1f5c4c6a0c33283d3cf01f", "self_ask", 2, "T29", 16, 7, 5),
    ("d5eae7f581069548be69a8ad72a36224", "inquirer_prompted", 1, "T1", 2, 16, 1),
    ("d7202954a2004cc6ca9a97b0709b4c2f", "inquirer_prompted", 0, "T27", 15, 5, 2),
    ("dacd96a1923465cdcdfeb8308af844a8", "inquirer_prompted", 2, "T26", 28, 3, 1),
    ("ddb3fe6bf6c93976e20a95f0f238f7e8", "self_ask", 1, "T2", 2, 16, 13),
    ("defc2da781690427dd0ec179a9976cca", "self_ask", 1, "T31", 13, 16, 2),
    ("e285bd3eab67013da5cbf217d47a22f7", "inquirer_prompted", 0, "T4", 6, 4, 0),
    ("e307d99409990c27b3012fabc7a4b6a6", "inquirer_prompted", 1, "T28", 2, 6, 1),
    ("e30b15ecd632a1f7128ce8491d0bfb2a", "self_ask", 1, "T30", 2, 16, 6),
    ("e362e19f75f4f6c8a037994f80cd7c91", "inquirer_prompted", 1, "T20", 6, 10, 1),
    ("e3726923050ca1d66a084f24e9361efe", "self_ask", 1, "T14", 13, 16, 23),
    ("e4735ac5a6aa5c87fe4b47cd3bd6821f", "self_ask", 0, "T21", 14, 2, 1),
    ("e47ab0f96bc328281733256ed3741217", "self_ask", 1, "T10", 21, 0, 3),
    ("e6830d25d4de1a6bda690d277f46246d", "inquirer_prompted", 0, "T26", 28, 4, 1),
    ("e6bdcbf5a5829ef0e49582e72b6c12e5", "inquirer_prompted", 0, "T31", 13, 10, 1),
    ("e6c674337dd48ebb64197f42df243b63", "inquirer_prompted", 2, "T13", 4, 6, 1),
    ("e6d40462db488a504ae3fb78a918879a", "self_ask", 2, "T8", 8, 8, 2),
    ("e6f2d6e30236b9131a5147d5858fbf44", "self_ask", 0, "T24", 25, 12, 2),
    ("e8aedca4ae425ed1bc868700e6d893c8", "inquirer_prompted", 2, "T12", 18, 5, 3),
    ("edc2ec008fd77348bd8280580ce1448a", "inquirer_prompted", 0, "T20", 6, 7, 1),
    ("eff7df36f53a1d39c1e546557320d4ec", "self_ask", 0, "T5", 16, 16, 1),
    ("f0dc539552103d69ff345d3afc33ea6f", "self_ask", 0, "T15", 4, 14, 9),
    ("f240219dcc83af13f06c28422066e28c", "inquirer_prompted", 0, "T24", 25, 10, 0),
    ("f2aea99d2976218d84187f83a9352903", "self_ask", 0, "T0", 16, 16, 12),
    ("f3c0266ff5f8155e92eb9310dfedc599", "self_ask", 0, "T28", 2, 16, 8),
    ("f6b86b38f00d8a6a6bb042898e052541", "inquirer_prompted", 2, "T3", 14, 7, 0),
    ("f830e0b66e0d3634276930ec12f09a8f", "inquirer_prompted", 1, "T4", 26, 7, 1),
    ("fa16337494a754c639fbf2847f6deb54", "self_ask", 0, "T12", 4, 9, 6),
    ("fcbca6518b75b88f1ea4f1ecf798df23", "inquirer_prompted", 0, "T7", 2, 14, 3),
    ("fd22ca14e7d0eefd42277ded5ac9efc5", "inquirer_prompted", 2, "T6", 34, 11, 3),
    ("fd573639a5505237504ad54bbc4f4d69", "inquirer_prompted", 1, "T31", 13, 7, 1),
    ("fe5fabf178d52b21cb22f0d4b37103d5", "inquirer_prompted", 2, "T19", 4, 16, 3),
    ("feab8c05feef86c4aa8d8168741366a6", "self_ask", 0, "T13", 4, 14, 4),
    ("ff7dc45121057e7dd350575cea411e81", "self_ask", 0, "T7", 2, 16, 13),
]


CAP = 16  # every one of the 204 runs; asserted below rather than trusted


class _RetailIdx:
    """Two readable records, addressed by key, standing in for `RetailIndex`.

    The replay charges calls; it does not need the campaign's real customer records, and must
    not carry them. What the fixture pins is HOW MANY ok calls each run's transcript held, and
    the charge is one per call whatever the call read. `uids_for_call` returning nothing for
    these arguments is therefore immaterial here, and the resolver is exercised for real by
    `tests/test_retail_evidence_wiring.py`.
    """

    uids = {"orders:#W1": "u-order-1"}
    item_to_product: dict[str, str] = {}
    titles = {"orders:#W1": "orders/#W1"}
    text_sha = {"orders:#W1": "a"}

    def uid(self, table, rid):
        return self.uids[f"{table}:{rid}"]


def _harvest_calls(n: int) -> list[dict]:
    """`n` successful transcript reads, in the shape `as_dict_call` hands the meter."""
    return [{"tool_name": "get_order_details", "kwargs_json": '{"order_id":"#W1"}', "ok": True}] * n


def _ledger_at_harvest(pre: int) -> BudgetLedger:
    """A ledger holding exactly the in-loop spend this run had reached when it finished."""
    led = BudgetLedger(cap=CAP)
    for _ in range(pre):
        led.charge_retrieval(1.0)
    return led


def _meter(led: BudgetLedger, n_post: int) -> tuple[tuple[str, ...], bool]:
    """`meter_env_calls`, normalised to `(uids, overrun)` WHATEVER SHAPE IT RETURNS.

    Deliberate, and the point of this helper: the subject of these tests is whether a finished
    run survives its own transcript, not the arity of the return value. Written the obvious way,
    the regression below failed at the pre-change commit with `ValueError: not enough values to
    unpack` -- a true failure for the wrong reason, indistinguishable from the test being
    aimed at a function that does not exist yet. Normalising here makes the pre-change failure
    the one that matters: `BudgetExceeded` on a run the campaign completed.
    """
    from pi_run.stages.tau2_runner import meter_env_calls

    got = meter_env_calls(_RetailIdx(), led, _harvest_calls(n_post))
    if isinstance(got, tuple) and len(got) == 2 and isinstance(got[1], bool):
        return got
    return tuple(got), False  # pre-change shape: uids only, and no way to report an overrun


def test_the_fixture_is_the_published_population() -> None:
    """Provenance, checked here so no assertion below rests on an unnamed set of runs."""
    ids = sorted(r[0] for r in _ROWS)
    assert len(_ROWS) == 204
    assert len(set(ids)) == 204
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16] == RUN_IDS_SHA
    assert Counter(r[1] for r in _ROWS) == {"inquirer_prompted": 102, "self_ask": 102}
    assert len({r[3] for r in _ROWS}) == 32, "32 recorded dialogues"
    assert len({(r[3], r[4]) for r in _ROWS}) == 34, "34 fork points"
    assert all(r[5] + r[6] >= 0 for r in _ROWS)


def test_the_fixture_could_have_come_out_otherwise() -> None:
    """Non-vacuity. If no recorded run had exceeded the cap, the regression below would pass for
    a reason with nothing to do with the fix, and a clean pass would be indistinguishable from
    an inert test."""
    assert len([r for r in _ROWS if r[5] + r[6] > CAP]) == 70, "over-cap runs are really present"
    assert all(r[5] <= CAP for r in _ROWS), "and no run had blown the cap in-loop by itself"
    assert any(r[6] == 0 for r in _ROWS), "while some charged nothing post-hoc at all"


def test_no_completed_run_is_deleted_by_its_own_transcript() -> None:
    """THE REGRESSION, and every figure the disclosure rests on.

    Each of these 204 runs finished with `status: ok` and a real reward. Pre-change the harvest
    raises `BudgetExceeded` on 70 of them, asymmetrically: 57 of the 102 control runs against 13
    of the 102 treatment runs, because `self_ask` charges 18.3 retrieval calls on average against
    a cap of 16 while `inquirer_prompted` charges 9.9. So the deletion selects on the endpoint --
    the control arm's expensive runs are its long dialogues, which are the evidence for the
    claim. 102 pairs collapse to 41 and 32 recorded dialogues to 19.
    """
    refused: list[tuple[str, str]] = []
    overruns: list[tuple[str, str]] = []
    for run_id, arm, _seed, _trace, _k, pre, n_post in _ROWS:
        led = _ledger_at_harvest(pre)
        try:
            _uids, overrun = _meter(led, n_post)
        except BudgetExceeded:
            refused.append((run_id, arm))
            continue
        if overrun:
            overruns.append((run_id, arm))
        assert led.spent.get("retrieval_calls", 0.0) == pre + n_post, (
            f"{run_id}: the spend is real and must be recorded whether or not it fits the cap"
        )

    by_arm = Counter(a for _r, a in refused)
    assert refused == [], (
        f"{len(refused)} of {len(_ROWS)} completed runs were deleted by their own harvest: "
        f"self_ask {by_arm['self_ask']}, inquirer_prompted {by_arm['inquirer_prompted']} -- "
        "the control arm loses over four times what the treatment arm loses, so the deletion "
        "runs in the published claim's favour"
    )

    over_by_arm = Counter(a for _r, a in overruns)
    assert len(overruns) == 70, "the overrun is reported on every run that had one, not swallowed"
    assert dict(over_by_arm) == {"self_ask": 57, "inquirer_prompted": 13}
    assert over_by_arm["self_ask"] > 4 * over_by_arm["inquirer_prompted"], "not a marginal skew"

    lost = {r for r, _a in overruns}
    survives: dict[tuple[str, int, int], bool] = {}
    for run_id, _arm, seed, trace, k, _pre, _n in _ROWS:
        key = (trace, k, seed)
        survives[key] = survives.get(key, True) and run_id not in lost
    assert len(survives) == 102, "102 pairs before"
    assert sum(survives.values()) == 41, "41 pairs after a re-run at the pre-change code"
    assert len({t for (t, _k, _s), ok in survives.items() if ok}) == 19, "19 of 32 dialogues left"


def test_the_in_loop_cap_still_refuses() -> None:
    """THE THING THAT MUST NOT MOVE. `retrieval_calls` is the one hard-capped currency and the
    in-loop cap genuinely truncates a rollout: there the policy is still choosing and a real
    overspend is still ahead of it, so letting one through would be worse than the defect this
    file is about. `charge_retrieval` and `check` are untouched.
    """
    led = BudgetLedger(cap=2)
    led.charge_retrieval(1.0)
    led.charge_retrieval(1.0)
    with pytest.raises(BudgetExceeded):
        led.charge_retrieval(1.0)
    assert led.spent["retrieval_calls"] == 2.0, "a refused charge is not recorded"
    led.check(0.0)
    with pytest.raises(BudgetExceeded):
        led.check(1.0)
