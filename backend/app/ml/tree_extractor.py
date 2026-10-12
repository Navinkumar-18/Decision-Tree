"""Extract a serializable decision-tree structure from a fitted scikit-learn estimator.

The extractor reads the real ``tree_`` attributes of the fitted estimator (node
count, children, splitting feature, threshold, impurity, samples and values) and
enriches each node with beginner-friendly display metadata:

- ``feature_display`` / ``feature_kind``: the original source column and whether
  the split is numerical or categorical (one-hot encoded).
- ``condition``: a readable question such as ``petal length (cm) <= 2.45`` or
  ``contract_type = Month-to-month``.
- ``threshold_display``: the threshold expressed in the original units of the
  source column (the scaler is inverted for display purposes).
- ``left_rule`` / ``right_rule`` / ``left_label`` / ``right_label``: the exact
  meaning of each outgoing branch.

Nothing here is fabricated: every value comes from the fitted estimator.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def format_number(value: float) -> str:
    """Format a number so thresholds stay short and readable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return str(v)
    # 6 significant digits removes float artefacts such as 46500.0004
    v = float(f"{v:.6g}")
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    text = f"{v:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else str(v)


def _feature_info(
    index: int,
    feature_names: List[str],
    feature_meta: Optional[Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Resolve a transformed feature index back to its source column."""
    meta = dict((feature_meta or {}).get(index, {}))
    name = feature_names[index] if 0 <= index < len(feature_names) else f"feature_{index}"
    if "source" not in meta:
        if name.startswith("num__"):
            meta["source"] = name[5:]
        elif name.startswith("cat__"):
            meta["source"] = name[5:]
        else:
            meta["source"] = name
    if "kind" not in meta:
        meta["kind"] = "categorical" if name.startswith("cat__") else "numeric"
    meta.setdefault("transformed_name", name)
    return meta


def _describe_split(
    threshold: float, meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Build human readable labels for a split node."""
    source = str(meta.get("source") or "feature")
    kind = meta.get("kind", "numeric")
    category = meta.get("category")

    if kind == "categorical" and category:
        # One-hot encoded column: value is 1 when the category matches.
        # value <= threshold (< 1) therefore means "category is NOT present".
        question = f"{source} = {category}"
        left_rule = f"{source} is not {category}"
        right_rule = question
        left_label, right_label = "No", "Yes"
        return {
            "condition": question,
            "left_rule": left_rule,
            "right_rule": right_rule,
            "left_label": left_label,
            "right_label": right_label,
            "threshold_display": float(threshold),
        }

    display_threshold = float(threshold)
    mean = meta.get("mean")
    scale = meta.get("scale")
    if kind == "numeric" and mean is not None and scale not in (None, 0):
        # Invert the StandardScaler so the shown threshold is in original units.
        display_threshold = float(threshold) * float(scale) + float(mean)
    formatted = format_number(display_threshold)
    try:
        rounded = float(formatted)
    except (TypeError, ValueError):
        rounded = display_threshold
    question = f"{source} <= {formatted}"
    return {
        "condition": question,
        "left_rule": question,
        "right_rule": f"{source} > {formatted}",
        "left_label": "Yes",
        "right_label": "No",
        "threshold_display": rounded,
    }


class TreeExtractor:
    @staticmethod
    def extract_tree(
        estimator,
        feature_names: List[str],
        target_names: Optional[List[str]] = None,
        feature_meta: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert the fitted estimator's ``tree_`` structure into JSON nodes."""
        tree = estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        impurity = tree.impurity
        n_node_samples = tree.n_node_samples
        value = tree.value

        n_nodes = int(tree.node_count)
        depths = [0] * n_nodes
        parents = [-1] * n_nodes
        is_classification = target_names is not None

        nodes: List[Dict[str, Any]] = []
        for i in range(n_nodes):
            left = int(children_left[i])
            right = int(children_right[i])
            is_leaf = left == right
            node: Dict[str, Any] = {
                "id": i,
                "type": "leaf" if is_leaf else "decision",
                "is_leaf": bool(is_leaf),
                "impurity": float(impurity[i]),
                "samples": int(n_node_samples[i]),
                "children_left": left if not is_leaf else -1,
                "children_right": right if not is_leaf else -1,
                "parent": parents[i],
                "depth": depths[i],
            }

            if not is_leaf:
                if 0 <= left < n_nodes:
                    depths[left] = depths[i] + 1
                    parents[left] = i
                if 0 <= right < n_nodes:
                    depths[right] = depths[i] + 1
                    parents[right] = i

                feat_idx = int(feature[i])
                meta = _feature_info(feat_idx, feature_names, feature_meta)
                labels = _describe_split(float(threshold[i]), meta)
                node.update(
                    {
                        "feature": meta.get("transformed_name"),
                        "feature_display": meta.get("source"),
                        "feature_kind": meta.get("kind", "numeric"),
                        "category": meta.get("category"),
                        "threshold": float(threshold[i]),
                        "threshold_display": labels["threshold_display"],
                        "condition": labels["condition"],
                        "left_rule": labels["left_rule"],
                        "right_rule": labels["right_rule"],
                        "left_label": labels["left_label"],
                        "right_label": labels["right_label"],
                    }
                )

            # Real class counts / regression value stored by scikit-learn.
            raw = np.asarray(value[i][0], dtype=float)
            if is_classification and target_names is not None and len(target_names) == len(raw):
                distribution = {str(target_names[j]): float(raw[j]) for j in range(len(raw))}
                best = int(np.argmax(raw))
                node["class_distribution"] = distribution
                node["prediction"] = str(target_names[best])
                node["prediction_value"] = float(raw[best])
            elif is_classification:
                best = int(np.argmax(raw)) if raw.size else 0
                node["class_distribution"] = {f"class_{j}": float(raw[j]) for j in range(len(raw))}
                node["prediction"] = f"class_{best}"
                node["prediction_value"] = float(raw[best]) if raw.size else 0.0
            else:
                node["class_distribution"] = None
                node["prediction"] = float(raw[0]) if raw.size else 0.0
                node["prediction_value"] = float(raw[0]) if raw.size else 0.0

            nodes.append(node)

        n_leaves = sum(1 for n in nodes if n["is_leaf"])
        return {
            "nodes": nodes,
            "n_nodes": n_nodes,
            "n_leaves": int(n_leaves),
            "max_depth": int(max(depths)) if nodes else 0,
        }
