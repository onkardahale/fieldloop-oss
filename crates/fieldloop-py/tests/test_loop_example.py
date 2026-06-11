"""Gate test for the canonical onboarding example (`examples/loop.py`).

The README's first-screen output is the onboarding promise, so it must not be allowed to
drift. This loads the real example module and asserts its stable summary counts — the
exact shape a new engineer sees on their first run. It pins counts, not confidence
decimals, so it is robust but still fails loudly if the loop stops doing what the README
says it does.
"""

import importlib.util
from pathlib import Path

_LOOP_PY = Path(__file__).resolve().parent.parent / "examples" / "loop.py"


def _load_loop():
    spec = importlib.util.spec_from_file_location("fieldloop_loop_example", _LOOP_PY)
    assert spec and spec.loader, "could not load examples/loop.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loop_example_matches_the_documented_first_run():
    loop = _load_loop()
    summary = loop.summarize(loop.run_loop())

    # The exact story the README promises on first run.
    assert summary == {
        "capture": {"rollouts": 3, "dropped": 0},
        "attribute": {"feedbacks": 3, "skipped": 0},
        "curate": {"accepted": 2, "needs_review": 1},
        "select_uploads": {"selected": 1, "dropped": 2},
        # The same engine run once more on the bundled real MCAP file.
        "mcap_demo": {"attributed": 4, "unattributed": 1},
    }


def test_loop_example_reconstructs_links_and_fails_closed():
    """The two onboarding 'aha' claims, asserted on the real output: outcomes with no
    rollout_id get reconstructed with confidence, and a low-confidence one is held out."""
    loop = _load_loop()
    result = loop.run_loop()

    feedbacks = result["attribute"]["feedbacks"]
    methods = sorted(fb["join_method"] for fb in feedbacks)
    assert methods == ["explicit", "temporal", "temporal"]
    # Every binding carries a scored confidence in [0, 1].
    assert all(0.0 <= fb["join_confidence"] <= 1.0 for fb in feedbacks)

    # Fail-closed: the sub-threshold binding is held out, with the honest reason.
    review = result["curate"]["needs_review"]
    assert len(review) == 1
    assert review[0]["reason"] == "below_min_confidence"

    # The safety (reflex) trigger is kept even at the zero budget.
    requests = result["select_uploads"]["requests"]
    assert len(requests) == 1
    assert requests[0]["tier"] == "reflex_a"
