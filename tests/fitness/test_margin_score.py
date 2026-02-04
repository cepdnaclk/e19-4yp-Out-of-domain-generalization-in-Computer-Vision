#!/usr/bin/env python3
"""
Quick sanity test for Margin Score implementation.
Tests edge cases and validates the mathematical properties.
"""

import numpy as np

from biomedxpro.utils.metrics import calculate_classification_metrics


def test_margin_score_perfect_predictions() -> None:
    """Test with perfect predictions - margin should be close to 1.0"""
    # Perfect predictions: [1, 0, 0] for class 0
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    y_prob = np.array(
        [
            [0.95, 0.03, 0.02],  # Confident class 0
            [0.02, 0.96, 0.02],  # Confident class 1
            [0.01, 0.02, 0.97],  # Confident class 2
            [0.93, 0.04, 0.03],  # Confident class 0
            [0.03, 0.94, 0.03],  # Confident class 1
        ]
    )

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print(f"✓ Perfect Predictions Test")
    print(f"  Margin Score: {metrics['margin_score']:.4f}")
    print(f"  Expected: Close to 0.92 (0.95-0.03 avg)")
    assert metrics["margin_score"] > 0.85, "High confidence should yield high margin"
    assert metrics["accuracy"] == 1.0, "All predictions correct"
    print()


def test_margin_score_confused_predictions() -> None:
    """Test with confused predictions - margin should be close to 0.0"""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])  # Technically correct
    y_prob = np.array(
        [
            [0.40, 0.35, 0.25],  # Barely class 0
            [0.30, 0.38, 0.32],  # Barely class 1
            [0.31, 0.30, 0.39],  # Barely class 2
        ]
    )

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print(f"✓ Confused Predictions Test")
    print(f"  Margin Score: {metrics['margin_score']:.4f}")
    print(f"  Expected: Close to 0.03-0.05 (weak margins)")
    assert -0.1 < metrics["margin_score"] < 0.15, "Confusion should yield low margin"
    print()


def test_margin_score_wrong_predictions() -> None:
    """Test with wrong predictions - margin should be negative"""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])  # All wrong
    y_prob = np.array(
        [
            [0.10, 0.80, 0.10],  # Predicted 1, actual 0
            [0.15, 0.15, 0.70],  # Predicted 2, actual 1
            [0.75, 0.15, 0.10],  # Predicted 0, actual 2
        ]
    )

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print(f"✓ Wrong Predictions Test")
    print(f"  Margin Score: {metrics['margin_score']:.4f}")
    print(f"  Expected: Negative (wrong predictions)")
    assert metrics["margin_score"] < 0, "Wrong predictions should have negative margin"
    assert metrics["accuracy"] == 0.0, "All predictions wrong"
    print()


def test_margin_score_mixed() -> None:
    """Test with mixed correct/incorrect predictions"""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 1])  # 50% correct
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],  # Correct, high confidence
            [0.20, 0.70, 0.10],  # Wrong, but confident in wrong
            [0.10, 0.80, 0.10],  # Correct, high confidence
            [0.60, 0.30, 0.10],  # Wrong
            [0.05, 0.10, 0.85],  # Correct, high confidence
            [0.10, 0.75, 0.15],  # Wrong
        ]
    )

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print(f"✓ Mixed Predictions Test")
    print(f"  Margin Score: {metrics['margin_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Expected: Moderate positive (mixed results)")
    # With 50% accuracy but varied confidence, margin should be moderate
    print()


def test_binary_classification() -> None:
    """Test that margin score works for binary classification too"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array(
        [
            [0.85, 0.15],
            [0.92, 0.08],
            [0.10, 0.90],
            [0.15, 0.85],
        ]
    )

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    print(f"  Margin Score: {metrics['margin_score']:.4f}")
    print(f"  Expected: ~0.77 (avg of 0.70, 0.84, 0.80, 0.70)")
    assert metrics["margin_score"] > 0.7, "Binary should work correctly"
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Margin Score Implementation")
    print("=" * 60)
    print()

    test_margin_score_perfect_predictions()
    test_margin_score_confused_predictions()
    test_margin_score_wrong_predictions()
    test_margin_score_mixed()
    test_binary_classification()

    print("=" * 60)
    print("✓ All Tests Passed!")
    print("=" * 60)
    print("=" * 60)
