import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.config import WebConfig
from web.database import PortalDatabase
from web.server import app, db, config


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "hardware" in data
    assert "operator_lock" in data


def test_examples_endpoint(client):
    res = client.get("/api/examples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 4
    assert any("Murnaghan" in ex["title"] for ex in data)
    assert any("Baillie" in ex["problem_id"] or "Lucas" in ex["title"] for ex in data)


def test_auth_login_with_invitation(client):
    # 1. Invalid invite code should fail
    bad_res = client.post(
        "/api/auth/login",
        json={"email": "researcher@oxford.ac.uk", "invite_code": "INVALID_CODE_123"},
    )
    assert bad_res.status_code == 401

    # 2. Master mathematician invite code should succeed
    good_res = client.post(
        "/api/auth/login",
        json={"email": "researcher@oxford.ac.uk", "invite_code": "PRIME_MATH_MATHEMATICIAN_2026"},
    )
    assert good_res.status_code == 200
    data = good_res.json()
    assert data["success"] is True
    assert data["role"] == "mathematician"
    assert "token" in data

    token = data["token"]

    # 3. Check /api/auth/me with bearer token
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "researcher@oxford.ac.uk"
    assert me_data["role"] == "mathematician"


def test_problem_submission_and_queue(client):
    # Authenticate first
    auth_res = client.post(
        "/api/auth/login",
        json={"email": "geometer@ias.edu", "invite_code": "PRIME_MATH_MATHEMATICIAN_2026"},
    )
    token = auth_res.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Test schema validation (missing statement)
    bad_payload = {
        "problem_id": "test-invalid-prob",
        "title": "Invalid Problem Without Statement",
        "field": "Combinatorics",
        "statement": "",
    }
    bad_submit = client.post("/api/problems/submit", json=bad_payload, headers=headers)
    assert bad_submit.status_code == 422

    # 2. Test valid problem submission
    valid_payload = {
        "problem_id": "test-partition-conjecture",
        "title": "Partition Character Modulo Invariance",
        "field": "Combinatorics & Number Theory",
        "context": "Let lambda be a staircase partition of size k.",
        "statement": r"Prove that \chi_\lambda(2^d, 1^z) \neq 0 for all maximal domino peelings.",
        "hypotheses": ["lambda is staircase", "d >= 0"],
        "max_iterations": 6,
    }
    submit_res = client.post("/api/problems/submit", json=valid_payload, headers=headers)
    assert submit_res.status_code == 200
    sub_data = submit_res.json()
    assert sub_data["problem_id"] == "test-partition-conjecture"
    assert "job_id" in sub_data
    assert sub_data["queue_position"] >= 1

    job_id = sub_data["job_id"]

    # 3. Inspect job detail
    job_res = client.get(f"/api/jobs/{job_id}", headers=headers)
    assert job_res.status_code == 200
    job_data = job_res.json()
    assert job_data["title"] == "Partition Character Modulo Invariance"
    assert job_data["status"] in ("queued", "held_operator_priority", "running")


def test_operator_priority_lock(client):
    # 1. Legacy FINANCE_ADMIN_2026 key must be rejected
    legacy_auth = client.post(
        "/api/auth/login",
        json={"email": "someone@example.com", "invite_code": "FINANCE_ADMIN_2026"},
    )
    assert legacy_auth.status_code == 401

    # 2. Non-admin domain does NOT have admin bypass (mathematician only with valid invite)
    fin_auth = client.post(
        "/api/auth/login",
        json={"email": "someone@example.com", "invite_code": "PRIME_MATH_MATHEMATICIAN_2026"},
    )
    assert fin_auth.status_code == 200
    assert fin_auth.json()["role"] == "mathematician"

    # 3. Configured admin emails from environment (.env) get admin role
    test_admins = config.admin_emails or ["test_operator_admin@example.org"]
    if not config.admin_emails:
        config.admin_emails = test_admins
        db.sync_admin_users(test_admins)

    for adm in test_admins:
        res = client.post(
            "/api/auth/login",
            json={"email": adm, "invite_code": "PRIME_MATH_ADMIN_2026"},
        )
        assert res.status_code == 200
        assert res.json()["role"] == "admin"

    # Admin login as the primary configured admin
    admin_auth = client.post(
        "/api/auth/login",
        json={"email": test_admins[0], "invite_code": "PRIME_MATH_ADMIN_2026"},
    )
    admin_token = admin_auth.json()["token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Lock queue as operator
    lock_res = client.post(
        "/api/admin/lock",
        json={"locked": True, "reason": "Running overnight TCS Benchmark Evaluation"},
        headers=admin_headers,
    )
    assert lock_res.status_code == 200
    assert lock_res.json()["operator_lock"] is True

    # 2. Check health endpoint reflects lock
    h_res = client.get("/api/health")
    assert h_res.json()["operator_lock"] is True
    assert "TCS Benchmark" in h_res.json()["operator_lock_reason"]

    # 3. Mathematician submits while locked -> must be placed in held state
    math_auth = client.post(
        "/api/auth/login",
        json={"email": "user@harvard.edu", "invite_code": "PRIME_MATH_MATHEMATICIAN_2026"},
    )
    math_headers = {"Authorization": f"Bearer {math_auth.json()['token']}"}

    held_payload = {
        "problem_id": "test-held-problem",
        "title": "Problem Submitted During Operator Run",
        "field": "Analysis",
        "statement": r"Prove convergence of \sum 1/n^2 to \pi^2 / 6.",
    }
    held_sub = client.post("/api/problems/submit", json=held_payload, headers=math_headers)
    assert held_sub.status_code == 200
    assert held_sub.json()["status"] == "held_operator_priority"
    assert held_sub.json()["operator_locked"] is True

    # 4. Release lock
    unlock_res = client.post(
        "/api/admin/lock",
        json={"locked": False, "reason": "Evaluations complete"},
        headers=admin_headers,
    )
    assert unlock_res.status_code == 200
    assert unlock_res.json()["operator_lock"] is False
