"""Classes to test a model on datasets.

"""
from typing import Any
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import logging
import pickle
from io import TextIOBase
from time import perf_counter
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from zensols.util.log import loglevel
from zensols.util.fail import APIError
from zensols.util import time
from zensols.persist import PersistableContainer, persisted
from zensols.config import Dictable
from .task import Task, TaskResponse, TaskDatasetFactory
from .instruct import InstructTaskRequest
from .metric import MetricsResult

logger = logging.getLogger(__name__)


class TestError(APIError):
    pass


@dataclass
class TestResult(PersistableContainer, Dictable):
    """Results from the a test run.

    """
    prediction_col: str = field()
    """Add a prediction column to add."""

    raw_col: bool = field()
    """The column of the raw model output to add or ``None`` to not add it."""

    time_elapsed: int = field()
    """Time in seconds it took to test."""

    predictions: tuple[dict[str, Any]] = field(repr=False)
    """The predictions as dict results, each as a row."""

    @property
    @persisted('_dataframe', transient=True)
    def dataframe(self) -> pd.DataFrame:
        """The dataframe representation of :obj:`predictions`."""
        return pd.DataFrame(self.predictions)

    # @property
    # @persisted('_metrics', transient=True)
    # def metrics(self) -> Metrics:
    #     metrics: MetricsResult = self.metrics_calculator.calculate(self.dataframe)

    def write_jsonl(self, writer: Path | TextIOBase):
        """Write predictions to a JSONL file or data sink."""
        pd.DataFrame(self.predictions).to_json(
            writer,
            orient='records',
            lines=True)


@dataclass
class Tester(Dictable):
    """Tests the fit of the model on a dataset.

    """
    source: TaskDatasetFactory = field()
    """A factory that creates new datasets used to evaluation."""

    task: Task = field()
    """The task used for to test the model."""

    result_file: Path = field()
    """The file to save the training statistics for benchmarking."""

    prediction_col: str = field(default='prediction')
    """Add a prediction column to add."""

    raw_col: bool = field(default=None)
    """The column of the raw model output to add or ``None`` to not add it."""

    result_mapper: Callable = field(default=lambda x: x)
    """Map the result by calling with the single :class:`.task.TaskResponse`."""

    limit: int | None = field(default=None)
    """The limit on the number of test cases to process."""

    def _test(self) -> Iterable[dict[str, Any]]:
        ds: Dataset = self.source.create()
        if self.limit is not None:
            ds = ds.select(range(min(len(ds), self.limit)))
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'testing {len(ds)} sample(s)...')
        for row in tqdm(ds, total=len(ds)):
            req = InstructTaskRequest(instruction=row['instruction'])
            res: TaskResponse = self.task.process(req)
            output: str = res.model_output
            res_row: dict[str, Any] = dict(row)
            res_row[self.prediction_col] = self.result_mapper(output)
            if self.raw_col is not None:
                res_row[self.raw_col] = res.model_output_raw
            yield res_row

    def test(self) -> tuple[dict[str, Any]]:
        """Run the tests and return the results with the predictions.  The
        prediction is added as with column (key) :obj:`prediction_col`.

        """
        res: tuple[dict[str, Any]]
        with loglevel('zensols.lmtask.generate', logging.WARNING):
            start: float = perf_counter()
            res = tuple(self._test())
            time_elapsed: float = perf_counter() - start
            if logger.isEnabledFor(logging.INFO):
                logger.info(time.format_elapse(
                    f'tested {len(res)} sample(s)', time_elapsed))
        return TestResult(
            prediction_col=self.prediction_col,
            raw_col=self.raw_col,
            time_elapsed=time_elapsed,
            predictions=res)

    @property
    def result_exists(self) -> bool:
        """Whether the trained model already exists."""
        return self.result_file.is_file()

    def save_result(self, result: TestResult):
        """Save the result to the file system."""
        result_path: Path = self.result_file
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f'wrote: {result_path}')

    def load_result(self) -> TestResult:
        """Load the model results."""
        path: Path = self.result_file
        if not path.is_file():
            raise TestError(
                f'It apperas the model exists but missing result file: {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)
