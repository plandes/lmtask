"""Classes to test a model on datasets.

"""
from typing import Any
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import logging
from zensols.util.log import loglevel
from zensols.util import time
from zensols.config import Dictable
from tqdm import tqdm
from datasets import Dataset
from .task import Task, TaskResponse, TaskDatasetFactory
from .instruct import InstructTaskRequest

logger = logging.getLogger(__name__)


@dataclass
class Tester(Dictable):
    """Tests the fit of the model on a dataset.

    """
    source: TaskDatasetFactory = field()
    """A factory that creates new datasets used to evaluation."""

    task: Task = field()
    """The task used for to test the model."""

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
            with time('tested {cnt} sample(s)', logger=logger):
                res = tuple(self._test())
                cnt = len(res)
        return res
