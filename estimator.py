from typing import List, Dict, Union, Sequence, Tuple
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def _is_nested(data: Union[List[int], List[List[int]]]) -> bool:
    """Check if data is nested (sequence labeling) or flat (classification)."""
    if len(data) == 0:
        return False
    return isinstance(data[0], (list, tuple, np.ndarray))

def _compute_accuracy_classification(
    predictions: Sequence[float],
    true_labels: Sequence[float],
    scores: Sequence[float],
    selected_indices: Sequence[int],
    plot: bool = False,
) -> float:
    assert np.isclose(np.sum(scores), 1), f"Scores sum: {np.sum(scores)}"

    R_PURE = 0.0
    pure_items = []
    for i in selected_indices:
        importance_weight = 1.0 / (len(scores) * scores[i] + 1e-8)
        R_PURE += (importance_weight * (predictions[i] == true_labels[i])) / len(selected_indices)
        pure_items.append(R_PURE)

    if plot:
        plt.plot(pure_items)
        plt.xlabel("Step")
        plt.ylabel("R_PURE")
        plt.title("Accuracy estimator trend")
        plt.savefig("accuracy_estimator_trend.png")

    return R_PURE


def _compute_precision_recall_classification(
    predictions: Sequence[float],
    true_labels: Sequence[float],
    scores: Sequence[float],
    selected_indices: Sequence[int],
    num_classes: int,
) -> Dict[str, float]:
    assert np.isclose(np.sum(scores), 1.0)

    N = len(scores)
    eps = 1e-8

    TP_hats = []

    for c in range(num_classes):
        tp_hat = 0.0
        for i in selected_indices:
            if predictions[i] == c and true_labels[i] == c:
                w_i = 1.0 / (N * scores[i] + eps)
                tp_hat += w_i

        PI_hat = sum((predictions[i] == c) for i in range(len(scores)))
        TI_hat = sum((true_labels[i] == c) for i in range(len(scores)))

        TP_hats.append((tp_hat, PI_hat, TI_hat))

    precisions = []
    recalls = []

    for tp_hat, PI_c, TI_c in TP_hats:
        precisions.append(tp_hat / (PI_c + eps))
        recalls.append(tp_hat / (TI_c + eps))

    return {
        "unbiased_precision": N / len(selected_indices) * np.mean(precisions),
        "unbiased_recall": N / len(selected_indices) * np.mean(recalls),
    }


def _compute_accuracy_sequence(
    predictions: List[List[int]],
    true_labels: List[List[int]],
    scores: List[float],
    selected_indices: List[int]
) -> float:
    N = len(scores)
    R_HT = 0.0

    for i in selected_indices:
        weight = 1.0 / (N * scores[i] + 1e-8)
        acc_seq = sum(predictions[i][j] == true_labels[i][j]
                      for j in range(len(true_labels[i]))) / len(true_labels[i])
        R_HT += weight * acc_seq

    R_HT /= len(selected_indices)
    return R_HT


def _compute_precision_recall_sequence(
    predictions: List[List[int]],
    true_labels: List[List[int]],
    scores: List[float],
    selected_indices: List[int],
    num_classes: int
) -> Dict[str, float]:
    N = len(scores)
    eps = 1e-8

    TP_hats = np.zeros(num_classes)
    PI_hats = np.zeros(num_classes)
    TI_hats = np.zeros(num_classes)

    for i in selected_indices:
        weight = 1.0 / (N * scores[i] + eps)
        for p, t in zip(predictions[i], true_labels[i]):
            if p == t:
                TP_hats[p] += weight

    for seq_pred, seq_true in zip(predictions, true_labels):
        for p in seq_pred:
            PI_hats[p] += 1
        for t in seq_true:
            TI_hats[t] += 1

    precisions = TP_hats / (PI_hats + eps)
    recalls = TP_hats / (TI_hats + eps)

    factor = N / len(selected_indices)
    
    return {
        "unbiased_precision": factor * np.mean(precisions),
        "unbiased_recall": factor * np.mean(recalls)
    }


def compute_accuracy_estimator(
    predictions: Union[List[int], List[List[int]]],
    true_labels: Union[List[int], List[List[int]]],
    scores: List[float],
    selected_indices: List[int],
    plot: bool = False,
) -> float:
    """
    Compute unbiased accuracy using Horvitz-Thompson estimator.
    Automatically detects classification (flat) vs sequence labeling (nested).
    """
    if _is_nested(predictions):
        return _compute_accuracy_sequence(predictions, true_labels, scores, selected_indices)
    else:
        return _compute_accuracy_classification(predictions, true_labels, scores, selected_indices, plot)


def compute_unbiased_precision_recall(
    predictions: Union[List[int], List[List[int]]],
    true_labels: Union[List[int], List[List[int]]],
    scores: List[float],
    selected_indices: List[int],
    num_classes: int
) -> Dict[str, float]:
    """
    Compute unbiased precision and recall using Horvitz-Thompson estimator.
    Automatically detects classification (flat) vs sequence labeling (nested).
    """
    if _is_nested(predictions):
        return _compute_precision_recall_sequence(predictions, true_labels, scores, selected_indices, num_classes)
    else:
        return _compute_precision_recall_classification(predictions, true_labels, scores, selected_indices, num_classes)


def compute_unbiased_metrics(
    predictions: Union[List[int], List[List[int]]],
    true_labels: Union[List[int], List[List[int]]],
    scores: List[float],
    selected_indices: List[int],
    num_classes: int
) -> Dict[str, float]:
    """
    Wrapper function to compute all unbiased metrics.
    Automatically detects classification (flat) vs sequence labeling (nested).
    """
    unbiased_acc = compute_accuracy_estimator(predictions, true_labels, scores, selected_indices)
    unbiased_pr_re = compute_unbiased_precision_recall(predictions, true_labels, scores, selected_indices, num_classes)

    return {
        "unbiased_accuracy": unbiased_acc,
        "unbiased_precision": unbiased_pr_re["unbiased_precision"],
        "unbiased_recall": unbiased_pr_re["unbiased_recall"],
    }

def _get_ngrams(text: str, n: int) -> Counter:
    tokens = text.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return Counter(ngrams)


def _compute_overlap_counts(
    prediction: str,
    reference: str,
    n: int
) -> Tuple[int, int, int]:
    """
    Returns:
        overlap_count,
        reference_ngram_count,
        prediction_ngram_count
    """
    pred_ngrams = _get_ngrams(prediction, n)
    ref_ngrams = _get_ngrams(reference, n)

    overlap = sum(
        min(pred_ngrams[g], ref_ngrams[g])
        for g in ref_ngrams
    )

    ref_total = sum(ref_ngrams.values())
    pred_total = sum(pred_ngrams.values())

    return overlap, ref_total, pred_total

def compute_unbiased_rouge_n(
    predictions: List[str],
    references: List[str],
    scores: List[float],
    selected_indices: List[int],
    n: int = 1
) -> Dict[str, float]:
    """
    Compute unbiased ROUGE-N (recall and precision) as average of
    per-sentence scores, using Horvitz-Thompson estimator.
    """
    assert np.isclose(np.sum(scores), 1.0)
    eps = 1e-8
    k = len(selected_indices)

    ht_recall = 0.0
    ht_precision = 0.0

    for i in selected_indices:
        overlap_i, ref_total_i, pred_total_i = _compute_overlap_counts(
            predictions[i],
            references[i],
            n
        )

        sentence_recall = overlap_i / (ref_total_i + eps)
        sentence_precision = overlap_i / (pred_total_i + eps)

        w_i = 1.0 / (k * scores[i] + eps)

        ht_recall += w_i * sentence_recall
        ht_precision += w_i * sentence_precision

    N = len(scores)
    unbiased_recall = ht_recall / N
    unbiased_precision = ht_precision / N

    return {
        f"unbiased_rouge_{n}_recall": unbiased_recall.item(),
        f"unbiased_rouge_{n}_precision": unbiased_precision.item(),
    }