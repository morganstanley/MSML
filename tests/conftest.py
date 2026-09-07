import pytest


@pytest.fixture(autouse=True)
def _restore_investigate_team_globals():
    """Save and restore it._one_call/_one_schema_call/record_finding per test.

    Tests monkeypatch `it._one_call`/`it._one_schema_call`/
    `InvestigatorSession.record_finding` directly (not via pytest's
    `monkeypatch`) — this fixture saves and restores all three around every
    test so one test's patch can't leak into another (it previously did:
    `TestRecordFindingCritic`'s permanent `record_finding` override broke
    `TestRecordTimeValidation` when run afterward).
    """
    from runcmp import investigate_team as it
    saved = (it._one_call, it._one_schema_call,
             it.InvestigatorSession.record_finding)
    yield
    (it._one_call, it._one_schema_call,
     it.InvestigatorSession.record_finding) = saved
