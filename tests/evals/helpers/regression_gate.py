"""Reusable regression gates for evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionResult:
    """Result of a single regression-threshold check."""

    name: str
    actual: float
    threshold: float

    @property
    def passed(self) -> bool:
        """Return True when the actual score meets the threshold."""

        return self.actual >= self.threshold


def assert_threshold(
    name: str,
    actual: float,
    threshold: float,
) -> RegressionResult:
    """Assert that a metric meets its minimum acceptable threshold."""

    result = RegressionResult(
        name=name,
        actual=actual,
        threshold=threshold,
    )

    if not result.passed:
        raise AssertionError(
            f"Regression gate failed for {name}: "
            f"actual={actual:.4f}, "
            f"required>={threshold:.4f}"
        )

    return result


def assert_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> list[RegressionResult]:
    """Validate multiple metrics against their thresholds."""

    results: list[RegressionResult] = []
    failures: list[str] = []

    for name, threshold in thresholds.items():
        if name not in metrics:
            raise KeyError(f"Metric {name!r} is missing from evaluation results.")

        result = RegressionResult(
            name=name,
            actual=metrics[name],
            threshold=threshold,
        )

        results.append(result)

        if not result.passed:
            failures.append(
                f"{name}: actual={result.actual:.4f}, required>={result.threshold:.4f}"
            )

    if failures:
        raise AssertionError(
            "Regression gates failed:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )

    return results


def print_regression_results(
    results: list[RegressionResult],
) -> None:
    """Print regression-gate results in CI-friendly format."""

    print("\nRegression gates:")

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(
            f"  [{status}] "
            f"{result.name}: "
            f"{result.actual:.4f} "
            f"(threshold={result.threshold:.4f})"
        )
