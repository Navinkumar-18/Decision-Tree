"""Predictions, decision paths and preprocessing behaviour."""

import numpy as np

from app.api import routes
from tests.conftest import upload_csv

IRIS_FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}


def _train_iris(client, **overrides):
    dataset = client.post("/api/datasets/sample/iris").json()
    body = {
        "dataset_id": dataset["dataset_id"],
        "target_column": "target",
        "algorithm": "decision_tree",
        "problem_type": "auto",
        "max_depth": 4,
    }
    body.update(overrides)
    response = client.post("/api/models/train", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def test_prediction_uses_fitted_model(client):
    model = _train_iris(client)
    response = client.post(
        f"/api/models/{model['model_id']}/predict", json={"features": IRIS_FEATURES}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in ("0", "1", "2")
    assert payload["target_column"] == "target"
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6

    # the API must agree with the stored estimator
    stored = routes.models[model["model_id"]]
    frame = routes._features_to_frame(stored, IRIS_FEATURES)
    expected = stored["estimator"].predict(stored["pipeline"].transform(frame))[0]
    assert payload["prediction"] == str(int(expected))


def test_prediction_matches_leaf_node(client):
    model = _train_iris(client)
    response = client.post(
        f"/api/models/{model['model_id']}/decision-path", json={"features": IRIS_FEATURES}
    )
    payload = response.json()
    nodes = {n["id"]: n for n in model["tree"]["nodes"]}
    leaf = nodes[payload["leaf_node_id"]]
    assert leaf["is_leaf"] is True
    assert payload["prediction"] == leaf["prediction"] == payload["tree_prediction"]


def test_prediction_requires_features(client):
    model = _train_iris(client)
    response = client.post(f"/api/models/{model['model_id']}/predict", json={"features": {}})
    assert response.status_code == 400


def test_missing_feature_values_are_imputed(client, numeric_dataset):
    model_response = client.post(
        "/api/models/train",
        json={"dataset_id": numeric_dataset["dataset_id"], "target_column": "label"},
    )
    model = model_response.json()
    response = client.post(
        f"/api/models/{model['model_id']}/predict",
        json={"features": {"length": 5.0, "width": ""}},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in ("0", "1")


def test_unknown_categorical_value_is_tolerated(client, categorical_dataset):
    model_response = client.post(
        "/api/models/train",
        json={
            "dataset_id": categorical_dataset["dataset_id"],
            "target_column": "churn",
            "max_depth": 3,
        },
    )
    model = model_response.json()
    response = client.post(
        f"/api/models/{model['model_id']}/predict",
        json={"features": {"age": 40, "income": 60000, "contract": "Unknown plan"}},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in ("yes", "no")


def test_regression_prediction(client):
    dataset = client.post("/api/datasets/sample/diabetes").json()
    train = client.post(
        "/api/models/train",
        json={
            "dataset_id": dataset["dataset_id"],
            "target_column": "target",
            "max_depth": 4,
        },
    ).json()
    features = {name: 0.0 for name in train["feature_columns"]}
    response = client.post(
        f"/api/models/{train['model_id']}/predict", json={"features": features}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["problem_type"] == "regression"
    assert isinstance(float(payload["prediction"]), float)


# ---------------------------------------------------------------------------
# Decision path
# ---------------------------------------------------------------------------
def test_decision_path_walks_the_actual_tree(client):
    model = _train_iris(client)
    response = client.post(
        f"/api/models/{model['model_id']}/decision-path", json={"features": IRIS_FEATURES}
    )
    assert response.status_code == 200
    payload = response.json()
    path = payload["path"]
    nodes = {n["id"]: n for n in model["tree"]["nodes"]}

    assert path[0] == 0
    assert nodes[path[-1]]["is_leaf"] is True
    assert len(payload["rules"]) == len(path) - 1

    for parent, child in zip(path, path[1:]):
        node = nodes[parent]
        assert child in (node["children_left"], node["children_right"])

    # independently recompute the path from the fitted estimator
    stored = routes.models[model["model_id"]]
    frame = routes._features_to_frame(stored, IRIS_FEATURES)
    transformed = stored["pipeline"].transform(frame)
    expected = sorted(int(n) for n in stored["estimator"].decision_path(transformed).indices)
    assert path == expected


def test_decision_path_rules_describe_real_splits(client):
    model = _train_iris(client)
    payload = client.post(
        f"/api/models/{model['model_id']}/decision-path", json={"features": IRIS_FEATURES}
    ).json()
    nodes = {n["id"]: n for n in model["tree"]["nodes"]}
    for rule, parent, child in zip(payload["rules"], payload["path"], payload["path"][1:]):
        node = nodes[parent]
        went_left = node["children_left"] == child
        assert rule["went_left"] == went_left
        assert rule["condition"] == (node["left_rule"] if went_left else node["right_rule"])


def test_categorical_decision_path(client, categorical_dataset):
    model_response = client.post(
        "/api/models/train",
        json={
            "dataset_id": categorical_dataset["dataset_id"],
            "target_column": "churn",
            "max_depth": 3,
        },
    )
    model = model_response.json()
    payload = client.post(
        f"/api/models/{model['model_id']}/decision-path",
        json={"features": {"age": 30, "income": 50000, "contract": "Month-to-month"}},
    ).json()
    nodes = {n["id"]: n for n in model["tree"]["nodes"]}
    assert payload["path"][0] == 0
    assert payload["prediction"] in ("yes", "no")
    for rule, parent in zip(payload["rules"], payload["path"]):
        # every rule must match the corresponding split node's own labels
        node = nodes[parent]
        assert rule["condition"] in (node.get("left_rule"), node.get("right_rule"))
        assert rule["feature"] in (
            node.get("feature_display"),
            node.get("feature"),
        )


def test_decision_path_requires_features(client):
    model = _train_iris(client)
    response = client.post(
        f"/api/models/{model['model_id']}/decision-path", json={"features": {}}
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Random forest predictions & paths
# ---------------------------------------------------------------------------
def test_random_forest_prediction_exposes_tree_votes(client):
    model = _train_iris(client, algorithm="random_forest", n_estimators=6, max_depth=4)
    payload = client.post(
        f"/api/models/{model['model_id']}/predict", json={"features": IRIS_FEATURES}
    ).json()
    assert payload["model_type"] == "random_forest"
    assert len(payload["tree_predictions"]) == 6
    assert sum(payload["tree_votes"].values()) == 6
    assert payload["prediction"] in payload["tree_votes"]
    assert "combine" in payload["note"] or "average" in payload["note"]


def test_random_forest_decision_path_for_selected_tree(client):
    model = _train_iris(client, algorithm="random_forest", n_estimators=4, max_depth=4)
    payload = client.post(
        f"/api/models/{model['model_id']}/decision-path",
        json={"features": IRIS_FEATURES, "tree_index": 2},
    ).json()
    assert payload["tree_index"] == 2
    assert payload["path"][0] == 0
    # ensemble prediction stays available alongside the single-tree explanation
    assert payload["prediction"]
    assert payload["tree_prediction"]

    # path must match that specific tree's decision_path
    stored = routes.models[model["model_id"]]
    frame = routes._features_to_frame(stored, IRIS_FEATURES)
    transformed = stored["pipeline"].transform(frame)
    sub_tree = stored["estimator"].estimators_[2]
    expected = sorted(int(n) for n in sub_tree.decision_path(transformed).indices)
    assert payload["path"] == expected


def test_random_forest_decision_path_rejects_bad_tree_index(client):
    model = _train_iris(client, algorithm="random_forest", n_estimators=3)
    response = client.post(
        f"/api/models/{model['model_id']}/decision-path",
        json={"features": IRIS_FEATURES, "tree_index": 17},
    )
    assert response.status_code == 404


def test_random_forest_regression_tree_outputs(client):
    dataset = client.post("/api/datasets/sample/diabetes").json()
    model = client.post(
        "/api/models/train",
        json={
            "dataset_id": dataset["dataset_id"],
            "target_column": "target",
            "algorithm": "random_forest",
            "n_estimators": 4,
            "max_depth": 4,
        },
    ).json()
    features = {name: 0.0 for name in model["feature_columns"]}
    payload = client.post(
        f"/api/models/{model['model_id']}/predict", json={"features": features}
    ).json()
    assert len(payload["tree_predictions"]) == 4
    assert all(isinstance(v, float) for v in payload["tree_predictions"])
    # ensemble prediction equals the mean of individual tree outputs
    # prediction labels are rounded for display, so allow a small tolerance
    assert abs(float(payload["prediction"]) - float(np.mean(payload["tree_predictions"]))) < 1e-3
