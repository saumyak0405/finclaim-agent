"""Optional integration test against a real model. Skipped unless GROQ_API_KEY is set."""
import os

import pytest

from evals.harness import main

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs GROQ_API_KEY")


def test_groq_smoke():
    assert main(["--provider", "groq", "--only", "t01_perf", "t05_injection", "--label", "pytest-groq"]) == 0
