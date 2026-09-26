import pytest


def test_a_fractional_float_in_an_integer_column_is_refused():
    """pyarrow TRUNCATES rather than raising: to_table stored 7.9 as 7 and said nothing, in a
    module whose docstring says it is "strict in both directions". A float reaching an integer
    column means a bug upstream -- a mean where a count belongs, a ratio where a turn index
    belongs -- and truncating it deletes the evidence."""
    import pyarrow as pa

    from pi_eval.schema import TABLES, SchemaViolation, to_table

    def row(**over):
        r = {}
        for f in TABLES["runs"]:
            if pa.types.is_integer(f.type):
                r[f.name] = 0
            elif pa.types.is_floating(f.type):
                r[f.name] = 0.0
            elif pa.types.is_boolean(f.type):
                r[f.name] = False
            elif pa.types.is_list(f.type):
                r[f.name] = []
            else:
                r[f.name] = ""
        r.update(over)
        return r

    with pytest.raises(SchemaViolation, match="fractional float"):
        to_table("runs", [row(max_turns=7.9)])

    # an EXACT float is fine: JSON has one number type, so 7.0 arrives for an int all the time
    assert to_table("runs", [row(max_turns=7.0)]).column("max_turns")[0].as_py() == 7
    assert to_table("runs", [row(max_turns=7)]).column("max_turns")[0].as_py() == 7
