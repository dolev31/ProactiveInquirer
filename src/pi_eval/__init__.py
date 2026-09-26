"""pi_eval — GOLD-ONLY. Nothing outside this package may import it.

Enforced three ways: an import-linter forbidden contract, a type wall (every field here is
gold_*-prefixed so it cannot collide with a TaskView field), and a process split — rollout
workers run with PI_GOLD_ROOT unset, so gold_root() raises and the raise IS the firewall
working.
"""
