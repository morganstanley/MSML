"""Tests for LLM client construction helpers (``alpha_lab.providers.utils.clients``).

Focus: ``_resolve_userid`` — the shared canonical-userid resolver behind the
``X-Ms-Assert-Username`` header, used by both the sync and async on-prem
OpenAI client paths.
"""

from __future__ import annotations

import sys
import types

from alpha_lab.providers.utils.clients import _resolve_userid


def test_resolve_userid_falls_back_to_user_env_when_ldap_unavailable(monkeypatch) -> None:
    # Normal off-prem case: ms.directory isn't importable, so the LDAP lookup
    # raises and we fall back to $USER.
    monkeypatch.setitem(sys.modules, "ms.directory", None)  # import -> raises
    monkeypatch.setenv("USER", "alice")
    assert _resolve_userid() == "alice"


def test_resolve_userid_defaults_to_unknown_when_user_unset(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ms.directory", None)
    monkeypatch.delenv("USER", raising=False)
    assert _resolve_userid() == "unknown"


def test_resolve_userid_returns_canonical_ldap_userid(monkeypatch) -> None:
    # Inject a fake ms.directory so the LDAP branch runs and resolves $USER to
    # the canonical userid — proving the header no longer carries bare $USER.
    monkeypatch.setenv("USER", "alice")

    class _FakePerson:
        userid = "canonical_alice"

    class _FakeConn:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def getProdID(self, _userid):
            return _FakePerson()

        def getUser(self, _userid):  # pragma: no cover - getProdID wins
            return _FakePerson()

    fake_mod = types.ModuleType("ms.directory")
    fake_mod.LDAPConnection = _FakeConn
    fake_mod.FWD_PROD_HOST = "fwd-prod"
    # Parent package must exist for ``from ms.directory import ...`` to resolve.
    monkeypatch.setitem(sys.modules, "ms", types.ModuleType("ms"))
    monkeypatch.setitem(sys.modules, "ms.directory", fake_mod)

    assert _resolve_userid() == "canonical_alice"


def test_resolve_userid_falls_back_when_ldap_returns_no_person(monkeypatch) -> None:
    # getProdID/getUser both return None -> ``None.userid`` raises -> fall back.
    monkeypatch.setenv("USER", "bob")

    class _FakeConn:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def getProdID(self, _userid):
            return None

        def getUser(self, _userid):
            return None

    fake_mod = types.ModuleType("ms.directory")
    fake_mod.LDAPConnection = _FakeConn
    fake_mod.FWD_PROD_HOST = "fwd-prod"
    monkeypatch.setitem(sys.modules, "ms", types.ModuleType("ms"))
    monkeypatch.setitem(sys.modules, "ms.directory", fake_mod)

    assert _resolve_userid() == "bob"
