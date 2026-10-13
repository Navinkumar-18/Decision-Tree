import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

CATEGORICAL_CSV = """age,income,contract,churn
25,40000,Month-to-month,yes
31,52000,One year,no
45,80000,Two year,no
23,,Month-to-month,yes
51,90000,One year,no
29,45000,,yes
60,70000,Two year,no
35,60000,Month-to-month,yes
41,,One year,no
27,48000,Two year,no
33,55000,Month-to-month,yes
47,72000,One year,no
52,85000,Two year,no
22,38000,Month-to-month,yes
58,66000,One year,no
"""

NUMERIC_CSV = """length,width,label
1.0,2.0,0
1.2,2.1,0
1.1,1.9,0
5.0,7.0,1
5.2,7.1,1
4.9,6.8,1
5.1,7.3,1
1.3,2.4,0
4.8,6.9,1
5.3,7.5,1
"""


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def upload_csv(client, content: str, filename: str = "data.csv"):
    response = client.post(
        "/api/datasets/upload",
        files={"file": (filename, io.BytesIO(content.encode()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def categorical_dataset(client):
    return upload_csv(client, CATEGORICAL_CSV, "people.csv")


@pytest.fixture()
def numeric_dataset(client):
    return upload_csv(client, NUMERIC_CSV, "numbers.csv")


@pytest.fixture()
def iris_dataset(client):
    response = client.post("/api/datasets/sample/iris")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def student_dataset(client):
    """The exact ``student_pass_classification.csv`` reported dataset."""
    path = os.path.join(os.path.dirname(__file__), "data", "student_pass_classification.csv")
    with open(path, "rb") as handle:
        content = handle.read()
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("student_pass_classification.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()
