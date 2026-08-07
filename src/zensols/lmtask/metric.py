"""Task-level benchmark metrics.

Metric semantics are intentionally separated from benchmark execution so each
task configuration can select the appropriate scorer (classification,
regression, multilabel, generation, etc.).

"""
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from zensols.config import Dictable


@dataclass
class Metric(Dictable):
    """A scalar benchmark metric."""

    name: str = field()
    """The metric name, such as ``accuracy``, ``precision``, ``recall`` or
    ``f1``.

    """
    value: float = field()
    """The computed scalar metric value."""

    average: str | None = field(default=None)
    """The averaging strategy, such as ``micro``, ``macro`` or ``weighted``;
    ``None`` when averaging does not apply.

    """


@dataclass
class ClassMetric(Dictable):
    """Metrics computed for a single class."""

    label: str = field()
    """The class label represented by this result."""

    precision: float = field()
    """Precision for the class."""

    recall: float = field()
    """Recall for the class."""

    f1: float = field()
    """F1 score for the class."""

    support: int = field()
    """The number of gold examples belonging to the class."""


@dataclass
class MetricsResult(Dictable):
    """Task-specific metrics returned by a :class:`MetricsCalculator`."""

    primary: Metric = field()
    """The metric considered the primary benchmark result for the task."""

    metrics: tuple[Metric, ...] = field(default=())
    """The aggregate scalar metrics computed for the task."""

    per_class: tuple[ClassMetric, ...] = field(default=())
    """Optional per-class metrics; empty when per-class reporting does not
    apply.

    """
    invalid_count: int = field(default=0)
    """The number of predictions that could not be interpreted as valid task
    outputs.

    """


@dataclass
class MetricsCalculator(Dictable, metaclass=ABCMeta):
    """Calculate task-level metrics from tester prediction rows."""

    label_column: str = field(default='label')
    """The dataframe column containing gold/reference values."""

    prediction_column: str = field(default='prediction')
    """The dataframe column containing model predictions."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> MetricsResult:
        """Return task-specific benchmark metrics.

        :param df: tester output containing gold labels and model predictions
        :return: the computed task metrics
        """


@dataclass
class ClassificationMetricsCalculator(MetricsCalculator):
    """Single-label classification metrics.
    """

    averages: tuple[str, ...] = field(
        default=('micro', 'macro', 'weighted'))
    """The averaging strategies used for aggregate precision, recall and F1
    metrics.

    """
    primary_average: str = field(default='macro')
    """The averaging strategy used for the primary metric."""

    primary_metric: str = field(default='f1')
    """The metric name used as the benchmark primary result."""

    labels: tuple[str, ...] | None = field(default=None)
    """The ordered valid class labels; ``None`` infers labels from the
    gold/reference column.

    """
    def calculate(self, df: pd.DataFrame) -> MetricsResult:
        y_true = df[self.label_column]
        y_pred = df[self.prediction_column]

        labels: tuple[str, ...]
        if self.labels is None:
            labels = tuple(sorted(set(y_true.dropna())))
        else:
            labels = self.labels

        valid = y_pred.isin(labels)
        invalid_count = int((~valid).sum())

        metrics = [
            Metric('accuracy', float(accuracy_score(y_true, y_pred)))
        ]

        for average in self.averages:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average=average,
                zero_division=0)
            metrics.extend((
                Metric('precision', float(precision), average),
                Metric('recall', float(recall), average),
                Metric('f1', float(f1), average),
            ))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0)

        per_class = tuple(
            ClassMetric(
                label=str(label),
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]))
            for i, label in enumerate(labels))

        primary = next(
            metric for metric in metrics
            if metric.name == self.primary_metric and
            metric.average == self.primary_average)

        return MetricsResult(
            primary=primary,
            metrics=tuple(metrics),
            per_class=per_class,
            invalid_count=invalid_count)
