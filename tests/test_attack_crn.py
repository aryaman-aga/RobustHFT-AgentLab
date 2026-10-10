"""Common random numbers for the attack, and frozen-attack replay.

Before this fix MisGen was seeded by the arm *name*, so three arms with identical
configs (injected__grounded_risk, injected__full_defense, injected__kappa_0.7)
reported ASR 25.3 / 22.7 / 31.3 -- attack noise as large as the defence effects.
"""
import pytest

from rhal.evaluation.experiment import RunSpec, Simulation


def _texts(tape):
    return {d: [it.text for it in items] for d, items in tape.docs.items()}


def test_arm_name_does_not_change_results(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    a = sim.run(RunSpec("injected__grounded_risk", "misgen", True, True, True))
    b = sim.run(RunSpec("injected__kappa_0.7", "misgen", True, True, True))
    assert a.metrics == b.metrics
    assert _texts(a.tape) == _texts(b.tape)


def test_replay_against_source_agent_reproduces_run(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    src = sim.run(RunSpec("injected__none", "misgen"))
    rep = sim.run(RunSpec("frozen__none", "replay"), tape=src.tape)
    assert rep.metrics == src.metrics
    assert [s["action"] for s in rep.treated_steps] == [s["action"] for s in src.treated_steps]
    assert rep.tape is None and rep.misgen_summary is None  # a replay records nothing new


def test_frozen_arms_face_identical_attack(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    tape = sim.run(RunSpec("injected__none", "misgen")).tape
    g = sim.run(RunSpec("frozen__grounded", "replay", True, False, True), tape=tape)
    gr = sim.run(RunSpec("frozen__grounded_risk", "replay", True, True, True), tape=tape)
    # same pool contents and same bias context every day, whatever the defence did
    assert [s["visible_synthetic_frac"] for s in g.treated_steps] == [s["visible_synthetic_frac"] for s in gr.treated_steps]
    assert [s["bias"] for s in g.treated_steps] == [s["bias"] for s in gr.treated_steps] == [tape.bias[s["date"]] for s in g.treated_steps]


def test_without_feedback_adaptive_arms_inject_identical_documents(small_cfg, small_dataset, detector):
    """Keyed per-(day, item) streams: different defences, same attack draws."""
    cfg = small_cfg.with_overrides({"misgen.use_feedback": False, "misgen.delta": 0.0})
    sim = Simulation(cfg, small_dataset, detector, 42)
    u = sim.run(RunSpec("u", "misgen", False, False, False))
    d = sim.run(RunSpec("d", "misgen", True, True, True))
    assert u.metrics["ASR"] != d.metrics["ASR"] or u.metrics["CR"] != d.metrics["CR"]  # agents really differ ...
    assert _texts(u.tape) == _texts(d.tape) and u.tape.bias == d.tape.bias  # ... the attack does not


def test_replay_requires_tape_and_only_for_replay(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    with pytest.raises(ValueError):
        sim.run(RunSpec("frozen__none", "replay"))
    tape = sim.run(RunSpec("injected__none", "misgen")).tape
    with pytest.raises(ValueError):
        sim.run(RunSpec("injected__none", "misgen"), tape=tape)
