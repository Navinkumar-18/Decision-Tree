"""Convert scikit-learn tree objects into the UI's serialisable node graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..preprocessing.preprocessor import PreparedFeatures


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "?"
    rounded = round(float(value))
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    text = f"{value:.4g}"
    return text


def _classes_for(estimator: Any) -> list[str] | None:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return None
    return [str(value) for value in classes]


def extract_tree(
    estimator: Any,
    prepared: PreparedFeatures,
    problem_type: str,
) -> dict[str, Any]:
    tree = estimator.tree_
    class_names = _classes_for(estimator) if problem_type == "classification" else None
    n_nodes = int(tree.node_count)

    depths = [-1] * n_nodes
    parents = [-1] * n_nodes

    def walk(node_id: int, parent: int, depth: int) -> None:
        if node_id < 0 or node_id >= n_nodes:
            return
        parents[node_id] = parent
        depths[node_id] = depth
        if tree.children_left[node_id] != tree.children_right[node_id]:
            walk(int(tree.children_left[node_id]), node_id, depth + 1)
            walk(int(tree.children_right[node_id]), node_id, depth + 1)

    walk(0, -1, 0)

    nodes: list[dict[str, Any]] = []
    for node_id in range(n_nodes):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        is_leaf = left == right
        weighted_samples = float(tree.weighted_n_node_samples[node_id])
        samples = int(round(weighted_samples))
        raw_values = np.asarray(tree.value[node_id][0], dtype=float)

        node: dict[str, Any] = {
            "id": node_id,
            "type": "leaf" if is_leaf else "decision",
            "is_leaf": is_leaf,
            "impurity": float(tree.impurity[node_id]),
            "samples": samples,
            "children_left": left,
            "children_right": right,
            "parent": int(parents[node_id]),
            "depth": int(depths[node_id]),
        }

        if problem_type == "classification":
            # Newer scikit-learn stores class proportions in tree.value while
            # older versions store raw counts. Normalise both to counts.
            if len(raw_values) and abs(float(raw_values.sum()) - 1.0) < 1e-6 and weighted_samples > 1:
                counts = [int(round(float(value) * weighted_samples)) for value in raw_values]
            else:
                counts = [int(round(float(value))) for value in raw_values]
            labels = class_names or [str(index) for index in range(len(counts))]
            distribution = {
                label: count for label, count in zip(labels, counts)
            }
            best = int(np.argmax(counts)) if len(counts) else 0
            node["class_distribution"] = distribution
            node["prediction"] = labels[best] if labels else None
            node["prediction_value"] = float(counts[best]) if len(counts) else 0.0
        else:
            value = float(raw_values[0]) if len(raw_values) else 0.0
            node["class_distribution"] = None
            node["prediction"] = value
            node["prediction_value"] = value

        if not is_leaf:
            feature_index = int(tree.feature[node_id])
            threshold = float(tree.threshold[node_id])
            meta = (
                prepared.metas[feature_index]
                if 0 <= feature_index < len(prepared.metas)
                else None
            )
            if meta is None:
                node["feature"] = f"feature_{feature_index}"
                node["feature_display"] = f"feature_{feature_index}"
                node["feature_kind"] = "numeric"
                node["condition"] = f"feature_{feature_index} <= {_format_number(threshold)}"
                node["left_rule"] = node["condition"]
                node["right_rule"] = f"feature_{feature_index} > {_format_number(threshold)}"
                node["left_label"] = "Yes"
                node["right_label"] = "No"
                node["threshold"] = threshold
                node["threshold_display"] = threshold
            elif meta.kind == "categorical":
                category = meta.category or ""
                node["feature"] = meta.column
                node["feature_display"] = meta.column
                node["feature_kind"] = "categorical"
                node["category"] = category
                node["condition"] = f"{meta.column} = {category}"
                node["left_rule"] = f"{meta.column} is not {category}"
                node["right_rule"] = f"{meta.column} = {category}"
                node["left_label"] = "No"
                node["right_label"] = "Yes"
                node["threshold"] = threshold
                node["threshold_display"] = 0.5
            else:
                original = prepared.original_threshold(feature_index, threshold)
                display = _format_number(original)
                node["feature"] = meta.column
                node["feature_display"] = meta.column
                node["feature_kind"] = "numeric"
                node["condition"] = f"{meta.column} <= {display}"
                node["left_rule"] = f"{meta.column} <= {display}"
                node["right_rule"] = f"{meta.column} > {display}"
                node["left_label"] = "Yes"
                node["right_label"] = "No"
                node["threshold"] = threshold
                node["threshold_display"] = original

        nodes.append(node)

    return {
        "nodes": nodes,
        "n_nodes": n_nodes,
        "n_leaves": int(tree.n_leaves),
        "max_depth": int(tree.max_depth),
    }


def tree_path(
    estimator: Any,
    tree_payload: dict[str, Any],
    encoded_row: np.ndarray,
) -> tuple[list[int], list[list[int]], list[dict[str, Any]], int]:
    """Return path node ids, edges, human-readable rules and the leaf id."""
    leaf_id = int(estimator.apply(encoded_row)[0])
    indicator = estimator.decision_path(encoded_row)
    path = [int(index) for index in indicator.indices.tolist()]

    by_id = {node["id"]: node for node in tree_payload["nodes"]}
    edges: list[list[int]] = []
    rules: list[dict[str, Any]] = []
    for position in range(len(path) - 1):
        current = path[position]
        following = path[position + 1]
        node = by_id.get(current, {})
        went_left = following == node.get("children_left")
        branch = node.get("left_label", "Yes") if went_left else node.get("right_label", "No")
        edges.append([current, following])
        rules.append(
            {
                "node_id": current,
                "feature": node.get("feature"),
                "condition": node.get("condition"),
                "branch": branch,
                "went_left": bool(went_left),
            }
        )

    return path, edges, rules, leaf_id
