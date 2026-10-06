from topocal.demo import run_demo


def test_demo_runs_all_paths() -> None:
    rows = run_demo()
    decisions = {row["decision"] for row in rows}
    assert "accept_neural" in decisions
    assert "fallback_solver" in decisions
