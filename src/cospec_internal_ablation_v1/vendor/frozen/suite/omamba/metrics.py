import numpy as np


def classification_metrics(labels, probabilities, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), num_classes) or not len(labels):
        raise ValueError("预测概率或标签形状错误。")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or
            not np.allclose(probabilities.sum(1), 1, atol=1e-4) or
            np.any(labels < 0) or np.any(labels >= num_classes)):
        raise ValueError("预测必须是有效概率，标签必须是 0..classes-1。")
    pred = probabilities.argmax(1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, pred), 1)
    support = matrix.sum(1)
    per_class = np.divide(matrix.diagonal(), support, out=np.zeros(num_classes), where=support > 0)
    recall = per_class
    precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(num_classes),
                          where=matrix.sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes),
                   where=precision + recall > 0)
    correct = pred == labels
    confidence = probabilities.max(1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    target = np.eye(num_classes)[labels]
    oa = float(correct.mean())
    expected = float((matrix.sum(0) * matrix.sum(1)).sum() / len(labels) ** 2)
    return {
        "oa_percent": 100 * oa, "aa_percent": 100 * float(per_class[support > 0].mean()),
        "macro_f1_present_classes": float(f1[support > 0].mean()),
        "kappa": (oa - expected) / (1 - expected) if expected < 1 else None,
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()),
        "brier": float(((probabilities - target) ** 2).sum(1).mean()), "ece_15_bins": float(ece),
        "high_confidence_error_fraction": float(((confidence >= 0.9) & ~correct).mean()),
        "per_class_accuracy_percent": [100 * float(x) if n else None for x, n in zip(per_class, support)],
        "class_support": support.tolist(), "confusion_matrix": matrix.tolist(), "count": len(labels),
    }
