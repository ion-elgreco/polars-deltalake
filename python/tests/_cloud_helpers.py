"""Shared helpers for cloud-backend integration tests.

Each `test_cloud_*.py` module is tagged with `pytest.mark.integration` and
`pytest.mark.skipif(not docker_available(), ...)`, so the entire suite
opts into Docker-backed emulators only when a daemon is actually reachable
— and users can filter with `pytest -m "not integration"` / `-m integration`
regardless.
"""

from __future__ import annotations


def docker_available() -> bool:
    """Return True iff a Docker daemon answers a ping."""
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False
