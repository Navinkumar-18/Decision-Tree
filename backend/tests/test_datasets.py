"""Dataset endpoints: health, upload, validation, samples and preview."""

import io

from tests.conftest import CATEGORICAL_CSV, upload_csv


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "in-memory" in payload["storage"]


def test_list_sample_datasets(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    samples = response.json()["samples"]
    ids = {s["id"] for s in samples}
    assert {"iris", "wine", "breast_cancer", "diabetes"} <= ids
    for sample in samples:
        assert sample["default_target"]


def test_sample_dataset_loads_real_data(client):
    response = client.post("/api/datasets/sample/iris")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 150
    assert payload["columns"] == 5
    assert "target" in payload["column_names"]
    assert len(payload["preview"]) > 0


def test_csv_upload_returns_metadata(client):
    payload = upload_csv(client, CATEGORICAL_CSV, "people.csv")
    assert payload["rows"] == 15
    assert payload["columns"] == 4
    assert payload["column_names"] == ["age", "income", "contract", "churn"]
    assert payload["column_kinds"]["age"] == "numeric"
    assert payload["column_kinds"]["contract"] == "categorical"
    assert payload["missing_values"]["income"] == 2
    assert payload["categories"]["contract"] == ["Month-to-month", "One year", "Two year"]
    assert len(payload["preview"]) <= 20
    assert payload["dataset_id"]


def test_csv_upload_sanitises_missing_values_in_preview(client):
    payload = upload_csv(client, CATEGORICAL_CSV, "people.csv")
    # JSON must be strictly parseable: NaN values must come back as null
    incomes = [row["income"] for row in payload["preview"]]
    assert any(v is None for v in incomes)


def test_upload_rejects_non_csv(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 400


def test_upload_rejects_malformed_csv(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("broken.csv", io.BytesIO(b"\x00\x01\x02,weird\n"), "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]


def test_upload_rejects_single_column_dataset(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("one.csv", io.BytesIO(b"only\n1\n2\n"), "text/csv")},
    )
    assert response.status_code == 400


def test_upload_rejects_header_only_dataset(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("head.csv", io.BytesIO(b"a,b,c\n"), "text/csv")},
    )
    assert response.status_code == 400


def test_upload_does_not_crash_server_on_garbage(client):
    for payload in (b"", b"\xff\xfe\xfd", b"a,b\n1,2\n"):
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("x.csv", io.BytesIO(payload), "text/csv")},
        )
        assert response.status_code in (200, 400, 413)
    assert client.get("/api/health").status_code == 200


def test_dataset_preview_endpoint(client):
    payload = upload_csv(client, CATEGORICAL_CSV, "people.csv")
    response = client.get(f"/api/datasets/{payload['dataset_id']}/preview?limit=5")
    assert response.status_code == 200
    preview = response.json()
    assert len(preview["preview"]) == 5
    assert preview["rows"] == 15
    assert preview["column_kinds"]["churn"] == "categorical"


def test_dataset_preview_unknown_id(client):
    assert client.get("/api/datasets/missing/preview").status_code == 404


def test_problem_type_detection_endpoint(client):
    payload = upload_csv(client, CATEGORICAL_CSV, "people.csv")
    dataset_id = payload["dataset_id"]
    churn = client.get(f"/api/datasets/{dataset_id}/problem-type?target_column=churn")
    assert churn.status_code == 200
    assert churn.json()["problem_type"] == "classification"
    age = client.get(f"/api/datasets/{dataset_id}/problem-type?target_column=age")
    assert age.json()["problem_type"] == "regression"
    missing = client.get(f"/api/datasets/{dataset_id}/problem-type?target_column=nope")
    assert missing.status_code == 404


def test_manual_dataset_endpoint(client):
    response = client.post(
        "/api/datasets/manual",
        json={
            "name": "manual.csv",
            "columns": ["x", "y", "label"],
            "data": [[1, 2, "a"], [3, 4, "b"], [5, 6, "a"], [7, 8, "b"]],
        },
    )
    assert response.status_code == 200
    assert response.json()["rows"] == 4
