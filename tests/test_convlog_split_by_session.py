"""A session must land whole on one side of the train/test wall.

Two decision points from one conversation share its task, its repository and its user's
habits. Letting them straddle the wall is the standard way a 'clean' split leaks, which is
exactly what `split_key(suite, task_id, template_id)` exists to prevent -- so the convlog
adapter's template_id IS the session id.
"""

from pinq.splitting import split_of
from pinq_adapters.convlog.suite import ConvlogSuite


def test_template_id_is_the_session_id():
    assert ConvlogSuite.template_id_of("abc123:7") == "abc123"
    assert ConvlogSuite.template_id_of("abc123:0") == "abc123"


def test_every_decision_point_of_a_session_gets_one_split():
    for sid in ("s-one", "s-two", "s-three", "s-four", "s-five"):
        splits = {
            split_of("convlog", f"{sid}:{i}", ConvlogSuite.template_id_of(f"{sid}:{i}"))
            for i in range(12)
        }
        assert len(splits) == 1, f"{sid} straddled the wall: {splits}"
