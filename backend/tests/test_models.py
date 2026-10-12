"""Model generation, tree extraction, evaluation and forest tree selection."""

import numpy as np

from app.api import routes
from app.services.trainer import detect_problem_type
from tests.conftest import upload_csv


def _train(client, dataset_id, **overrides):
    body = {
        "dataset_id": dataset_id,
        "target_column": "target",
        "algorithm": "decision_tree",
        "problem_type": "auto",
    }
    body.update(overrides)
    response = client.post("/api/models/train", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Problem type detection
# ---------------------------------------------------------------------------
def test_detect_problem_type_numeric_targets():
    import pandas as pd

    assert detect_problem_type(pd.Series([0, 1, 0, 1, 1, 0])) == "classification"
    assert detect_problem_type(pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])) == "classification"
    continuous = pd.Series(np.linspace(0, 100, 200))
    assert detect_problem_type(continuous) == "regression"
    strings = pd.Series(["cat", "dog", "cat", "bird"])
    assert detect_problem_type(strings) == "classification"


# ---------------------------------------------------------------------------
# Decision tree - classification / regression
# ---------------------------------------------------------------------------
def test_decision_tree_classification(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"], max_depth=3)
    assert model["type"] == "decision_tree"
    assert model["problem_type"] == "classification"
    assert model["problem_type_detected"] == "classification"
    assert 0.0 <= model["metrics"]["accuracy"] <= 1.0
    assert model["metrics"]["accuracy"] >= 0.8
    assert set(model["metrics"]) >= {"accuracy", "precision", "recall", "f1_score", "confusion_matrix"}
    assert model["tree"]["n_nodes"] > 1
    assert model["tree"]["n_leaves"] >= 2
    assert model["tree"]["max_depth"] <= 3


def test_decision_tree_regression(client):
    response = client.post("/api/datasets/sample/diabetes")
    dataset_id = response.json()["dataset_id"]
    model = _train(client, dataset_id, max_depth=4)
    assert model["problem_type"] == "regression"
    metrics = model["metrics"]
    assert set(metrics) >= {"mae", "rmse", "r2_score"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert model["tree"]["nodes"][0]["is_leaf"] is False


def test_random_forest_classification(client, iris_dataset):
    model = _train(
        client, iris_dataset["dataset_id"], algorithm="random_forest", n_estimators=5, max_depth=3
    )
    assert model["type"] == "random_forest"
    assert model["n_estimators"] == 5
    assert len(model["trees"]) == 5
    assert model["metrics"]["accuracy"] >= 0.8
    for tree in model["trees"]:
        assert tree["n_nodes"] >= 1
        assert tree["nodes"][0]["id"] == 0


def test_random_forest_regression(client):
    response = client.post("/api/datasets/sample/diabetes")
    dataset_id = response.json()["dataset_id"]
    model = _train(
        client, dataset_id, algorithm="random_forest", n_estimators=4, max_depth=4
    )
    assert model["type"] == "random_forest"
    assert model["problem_type"] == "regression"
    assert len(model["trees"]) == 4
    assert model["metrics"]["rmse"] >= 0


# ---------------------------------------------------------------------------
# Tree structure extraction matches the fitted estimator
# ---------------------------------------------------------------------------
def test_extracted_tree_matches_fitted_estimator(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"], max_depth=4)
    stored = routes.models[model["model_id"]]
    estimator = stored["estimator"]
    tree = estimator.tree_
    nodes = model["tree"]["nodes"]

    assert len(nodes) == tree.node_count
    for i, node in enumerate(nodes):
        assert node["id"] == i
        assert node["samples"] == int(tree.n_node_samples[i])
        assert abs(node["impurity"] - float(tree.impurity[i])) < 1e-9
        if tree.children_left[i] == tree.children_right[i]:
            assert node["is_leaf"] is True
            assert node["children_left"] == -1
            expected = str(estimator.classes_[int(np.argmax(tree.value[i][0]))])
            assert node["prediction"] == expected
        else:
            assert node["is_leaf"] is False
            assert node["children_left"] == int(tree.children_left[i])
            assert node["children_right"] == int(tree.children_right[i])
            assert node["threshold"] == float(tree.threshold[i])
            feature_index = int(tree.feature[i])
            assert node["feature"] == stored["feature_names"][feature_index]

    # parent/child links are consistent
    for node in nodes:
        if node["children_left"] >= 0:
            assert nodes[node["children_left"]]["parent"] == node["id"]
            assert nodes[node["children_left"]]["depth"] == node["depth"] + 1
        if node["children_right"] >= 0:
            assert nodes[node["children_right"]]["parent"] == node["id"]


def test_numeric_thresholds_displayed_in_original_units(client, numeric_dataset):
    model = _train(client, numeric_dataset["dataset_id"], target_column="label", max_depth=2)
    root = model["tree"]["nodes"][0]
    assert root["feature_kind"] == "numeric"
    # The raw threshold is in scaled space; the display value must be in the
    # original range of the feature (lengths are ~1-5 in the fixture).
    assert 0.5 <= root["threshold_display"] <= 6.0
    assert "<=" in root["condition"]
    assert str(round(root["threshold_display"], 2)) in root["condition"]


def test_categorical_split_uses_readable_rules(client, categorical_dataset):
    model = _train(client, categorical_dataset["dataset_id"], target_column="churn", max_depth=3)
    split_nodes = [n for n in model["tree"]["nodes"] if not n["is_leaf"]]
    categorical_nodes = [n for n in split_nodes if n["feature_kind"] == "categorical"]
    assert categorical_nodes, "expected a split on the categorical column"
    node = categorical_nodes[0]
    assert node["feature_display"] == "contract"
    assert node["condition"] == "contract = Month-to-month"
    assert node["left_rule"] == "contract is not Month-to-month"
    assert node["right_rule"] == "contract = Month-to-month"
    # one-hot semantics: left = category absent (No), right = present (Yes)
    assert node["left_label"] == "No"
    assert node["right_label"] == "Yes"


# ---------------------------------------------------------------------------
# Validation & errors
# ---------------------------------------------------------------------------
def test_unknown_target_column_is_rejected(client, iris_dataset):
    response = client.post(
        "/api/models/train",
        json={"dataset_id": iris_dataset["dataset_id"], "target_column": "missing_column"},
    )
    assert response.status_code == 400
    assert "missing_column" in response.json()["detail"]


def test_single_value_target_is_rejected(client):
    csv = "a,b,c\n1,2,7\n3,4,7\n5,6,7\n7,8,7\n"
    dataset = upload_csv(client, csv, "constant.csv")
    response = client.post(
        "/api/models/train",
        json={"dataset_id": dataset["dataset_id"], "target_column": "c", "problem_type": "classification"},
    )
    assert response.status_code == 400
    assert "one unique value" in response.json()["detail"]


def test_unknown_dataset_is_rejected(client):
    response = client.post(
        "/api/models/train", json={"dataset_id": "nope", "target_column": "y"}
    )
    assert response.status_code == 404


def test_invalid_model_configuration_is_rejected(client, iris_dataset):
    response = client.post(
        "/api/models/train",
        json={
            "dataset_id": iris_dataset["dataset_id"],
            "target_column": "target",
            "algorithm": "random_forest",
            "n_estimators": 99999,
        },
    )
    assert response.status_code == 400


def test_problem_type_override_is_honoured(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"], problem_type="regression", max_depth=3)
    assert model["problem_type"] == "regression"
    assert "mae" in model["metrics"]


def test_small_dataset_falls_back_to_training_evaluation(client):
    csv = "a,b,y\n1,2,0\n2,3,1\n3,4,0\n"
    dataset = upload_csv(client, csv, "tiny.csv")
    model = _train(client, dataset["dataset_id"], target_column="y")
    assert model["eval_on"] == "train"
    assert model["eval_note"]
    assert model["n_test"] == 0
    assert model["metrics"]["accuracy"] is not None


def test_missing_target_values_are_dropped(client):
    csv = "a,b,y\n1,2,0\n2,3,1\n3,4,\n4,5,1\n5,6,0\n6,7,1\n7,8,0\n8,9,1\n"
    dataset = upload_csv(client, csv, "missing_target.csv")
    model = _train(client, dataset["dataset_id"], target_column="y")
    assert model["n_samples"] == 7


# ---------------------------------------------------------------------------
# Summary & feature importance
# ---------------------------------------------------------------------------
def test_model_summary_excludes_fitted_objects(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"])
    response = client.get(f"/api/models/{model['model_id']}/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline" not in payload
    assert "estimator" not in payload
    assert "tree" not in payload
    assert payload["tree_overview"]["nodes"] == model["tree"]["n_nodes"]
    assert payload["target_column"] == "target"


def test_feature_importance_endpoint(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"])
    response = client.get(f"/api/models/{model['model_id']}/feature-importance")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["feature_names"]) == 4
    assert abs(sum(payload["feature_importances"]) - 1.0) < 1e-6


def test_tree_endpoint_rejects_forest_models(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"], algorithm="random_forest", n_estimators=3)
    assert client.get(f"/api/models/{model['model_id']}/tree").status_code == 400


# ---------------------------------------------------------------------------
# Random forest tree selection
# ---------------------------------------------------------------------------
def test_forest_tree_listing_and_selection(client, iris_dataset):
    model = _train(
        client, iris_dataset["dataset_id"], algorithm="random_forest", n_estimators=4, max_depth=3
    )
    model_id = model["model_id"]

    listing = client.get(f"/api/models/{model_id}/forest/trees").json()
    assert listing["n_estimators"] == 4
    assert listing["trees"] == [0, 1, 2, 3]
    assert len(listing["overview"]) == 4
    assert all(o["depth"] >= 1 for o in listing["overview"])

    for index in range(4):
        response = client.get(f"/api/models/{model_id}/forest/trees/{index}")
        assert response.status_code == 200
        tree = response.json()
        assert tree["n_nodes"] == model["trees"][index]["n_nodes"]
        assert tree["nodes"][0]["id"] == 0

    assert client.get(f"/api/models/{model_id}/forest/trees/99").status_code == 404


def test_forest_endpoints_reject_decision_tree_models(client, iris_dataset):
    model = _train(client, iris_dataset["dataset_id"])
    assert client.get(f"/api/models/{model['model_id']}/forest/trees").status_code == 400


def test_unknown_model_endpoints_return_404(client):
    assert client.get("/api/models/missing/summary").status_code == 404
    assert client.get("/api/models/missing/tree").status_code == 404
