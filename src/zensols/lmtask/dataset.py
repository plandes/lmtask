"""An implementation of a dataset generator :class:`.task.TaskDatasetFactory`.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import pandas as pd
import datasets
from datasets import Dataset
from zensols.util import WritableContext
from zensols.persist import Stash
from .task import TaskDatasetFactoryError, TaskDatasetFactory


@dataclass
class LoadedTaskDatasetFactory(TaskDatasetFactory):
    """A utility class meant to be created from an application configuration.
    This class creates a dataframe used by :class:`.Trainer` and optionally does
    post processing (i.e. filtering and mapping).

    """
    source: Union[str, Path, Stash, pd.DataFrame, Dataset] = field(default=None)
    """Used as the source data in the created dataset."""

    load_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments given to :func:`datasets.load_dataset`."""

    pre_process: Union[str, Callable] = field(default=None)
    """Code to call after the dataset is created but before the task applies any
    template.  If this is a string :func:`exec` is used to evaluate it.
    Otherwise it is treated as a callable where the old dataset is the input and
    the returned value is the replaced dataset.

    """
    post_process: Union[str, Callable] = field(default=None)
    """Code to call after the dataset is created and the task has applied any
    template.

    :see: :obj:`pre_process`

    """
    @staticmethod
    def clear_generator_cache():
        path = Path('~/.cache/huggingface/datasets/generator')
        path = path.expanduser().absolute()
        if path.is_dir():
            shutil.rmtree(path)

    def _pre_process(self, ds: Dataset) -> Dataset:
        if isinstance(self.pre_process, str):
            _locs = locals()
            exec(self.pre_process)
            ds = _locs['ds']
        else:
            ds = self.pre_process(ds)
        return ds

    def _post_process(self, ds: Dataset) -> Dataset:
        if isinstance(self.post_process, str):
            _locs = locals()
            exec(self.post_process)
            ds = _locs['ds']
        else:
            ds = self.post_process(ds)
        return ds

    def _create(self) -> Dataset:
        """Instantiate using the :obj:`source` and :obj:`load_args`."""
        return self._from_source(self.source)

    def _from_source(self, source: Any) -> Dataset:
        ds: Dataset
        # create the dataset based on source type
        if isinstance(source, Dataset):
            ds = source
        elif source is None or isinstance(source, str):
            params: Dict[str, Any] = {}
            if source is not None:
                params['path'] = source
            params.update(self.load_args)
            ds = datasets.load_dataset(**params)
        elif isinstance(source, Path):
            files: Union[str, Tuple[str, ...]]
            if source.is_dir():
                files = tuple(map(str, source.iterdir()))
            else:
                files = str(source)
            if len(files) == 1 and \
               files[0].is_file() and \
               files[0].suffix == '.arrow':
                ds = Dataset.from_file(files[0], **self.load_args)
            else:
                args: Dict[str, Any] = dict(
                    # path tells load_dataset to load the files as text
                    path='text',
                    data_files=files,
                    split='train')
                args.update(self.load_args)
                ds = datasets.load_dataset(**args)
        elif isinstance(source, Stash):
            rows: List[Tuple[Any, ...]] = []
            stash: Stash = self.source
            key: str
            item: Any
            for key, item in stash.items():
                rows.append((key, item))
            df = pd.DataFrame(rows, columns='key item'.split())
            ds = self._from_source(df)
        elif isinstance(source, pd.DataFrame):
            ds = Dataset.from_pandas(source, **self.load_args)
        else:
            raise TaskDatasetFactoryError(
                f'Unknown source type: {type(source)}')
        return ds

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        # avoid pickling large stashes
        use_obj: bool = isinstance(self.source, (str, Path))
        return {'source': self.source if use_obj else type(self.source),
                'load_args': self.load_args,
                'pre_process': self.pre_process,
                'post_process': self.post_process}

    def _write(self, c: WritableContext):
        super()._write(c)
        c(self.source, 'source')
        c(self.load_args, 'load_args')
        for attr in 'pre_process post_process'.split():
            val = getattr(self, attr)
            c(val, attr) if val is None else c(val, attr, 'block')
