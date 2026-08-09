"""Reproducible LMTask benchmark execution and reporting.

The benchmark JSON file is the source of truth.  Markdown is rendered from that
structured result so reports can be regenerated without retraining a model.

"""
from __future__ import annotations
from typing import Any, ClassVar
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from collections import Counter
from datetime import datetime
from pathlib import Path
from io import StringIO
import importlib.metadata as metadata
import json
import logging
import os
import platform
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from zensols.config import Dictable
from zensols.util.executor import Executor
from zensols.persist import persisted, PersistedWork
from .torchconfig import CudaInfo
from .train import Trainer, TrainResult
from .test import Tester, TestResult
from .metric import MetricsCalculator, MetricsResult

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit(Dictable):
    """Dataset split size and optional class distribution."""

    name: str = field()
    """Logical split name, such as ``train``, ``validation`` or ``test``."""

    examples: int = field()
    """The number of examples in the split."""

    class_counts: dict[str, int] | None = field(default=None)
    """The class label counts; ``None`` when class distribution does not apply
    or was not requested.

    """


@dataclass
class DatasetSplitsResult(Dictable):
    """Contains the dataset splits."""
    splits: tuple[DatasetSplit] = field()

    def __iter__(self) -> Iterable[DatasetSplit]:
        return iter(self.splits)


@dataclass
class GitResult(Dictable):
    """Repository revision used for the benchmark."""

    commit: str = field()
    """The full Git commit SHA."""

    describe: str = field()
    """The human-readable revision returned by ``git describe``."""

    dirty: bool = field()
    """Whether the repository contained uncommitted or untracked changes when
    the benchmark was generated.

    """


@dataclass
class GpuResult(Dictable):
    """One CUDA device visible to the benchmark process."""

    index: int = field()
    """The CUDA device index visible to the current process."""

    name: str = field()

    total_memory: int = field()
    """The total device memory in bytes."""

    peak_allocated: int | None = field(default=None)
    """The peak memory allocated by PyTorch during benchmark execution in bytes;
    ``None`` when unavailable.

    """
    peak_reserved: int | None = field(default=None)
    """The peak memory reserved by the PyTorch CUDA allocator during benchmark
    execution in bytes; ``None`` when unavailable.

    """


@dataclass
class EnvironmentResult(Dictable):
    """Software and hardware environment used for the benchmark.
    """
    python: str = field()
    """The Python interpreter version."""

    os: str = field()
    """The operating-system description."""

    kernel: str = field()
    """The operating-system kernel version."""

    cuda_visible_devices: str | None = field()
    """The value of ``CUDA_VISIBLE_DEVICES``; ``None`` when not set."""

    cuda_runtime: str | None = field()
    """The CUDA runtime version used by the installed PyTorch build; ``None``
    when CUDA is unavailable.

    """
    nvidia_driver: str | None = field()
    """The NVIDIA driver version reported by ``nvidia-smi``; ``None`` when
    unavailable.

    """
    packages: dict[str, str] = field(default_factory=dict)
    """The relevant Python package names mapped to exact installed versions."""

    gpus: tuple[GpuResult, ...] = field(default=())
    """The CUDA devices visible to the benchmark process."""


@dataclass
class TrainingResult(Dictable):
    """Training facts captured from LMTask and the generated adapter."""

    elapsed_seconds: float = field()
    """The elapsed time required to run held-out task inference."""

    global_step: int = field()
    """The final trainer global step."""

    training_loss: float = field()
    """The final training loss reported by the trainer."""

    metrics: dict[str, Any] = field()
    """The task-specific benchmark metrics computed from tester predictions."""

    result_dir: str = field()
    """The directory where JSON, JSONL and Markdown benchmark artifacts are
    written.

    """
    adapter_size: int | None = field(default=None)
    """The size in bytes of ``adapter_model.safetensors``; ``None`` when
    unavailable.

    """


@dataclass
class TestingResult(Dictable):
    """Held-out task testing facts.

    """
    test_result: TestResult = field()
    """Result from :class:`.test.Tester`."""

    predictions_file: Path = field()
    """The path to the predictions JSONL file."""

    @property
    def elapsed_seconds(self) -> float:
        """Time taken to evaluate the test set."""
        return self.test_result.time_elapsed

    @property
    def examples(self) -> int:
        """Nummber of examples in the test set."""
        return len(self.test_result.predictions)


@dataclass
class BenchmarkResult(Dictable):
    """Complete machine-readable benchmark record.
    """
    name: str = field()
    """Name of this benchmark."""

    task_name: str = field()
    """The human-readable task or dataset name used in reports."""

    model_name: str = field()
    """The human-readable model name used in reports."""

    config_file: str = field()
    """The LMTask configuration file used for the benchmark."""

    created: str = field()
    """The timezone-aware ISO-8601 timestamp recording when the benchmark result
    was created.

    """
    git: GitResult = field()
    """The LMTask repository revision and working-tree state."""

    environment: EnvironmentResult = field()
    """The software, CUDA and hardware environment information."""

    datasets: tuple[DatasetSplit, ...] = field()
    """The train, validation and held-out test split metadata."""

    training: TrainingResult = field()
    """The model-training diagnostics and persisted adapter metadata."""

    testing: TestingResult = field()
    """The held-out task-inference metadata."""

    metrics: MetricsResult = field()

    notes: tuple[str, ...] = field(default=())
    """Optional free-form benchmark annotations."""


@dataclass
class BenchmarkRunner(Dictable):
    """Train, test, score and render one configured LMTask benchmark."""

    _DICTABLE_ATTRIBUTES = {'datasets'}
    _PACKAGE_NAMES: ClassVar[list[str]] = (
        'zensols.lmtask',
        'scikit-learn',
        'torch',
        'transformers',
        'trl',
        'peft',
        'accelerate',
        'datasets',
        'pandas',
        'pyarrow',
        'huggingface-hub')

    name: str = field()
    """Name of the runner."""

    task_name: str = field()
    """Name of the task."""

    model_name: str = field()
    """Name of the model used for training/testing."""

    config_file: Path = field()
    """The configuration file used for training and testing."""

    trainer: Trainer = field(repr=False)
    """The configured LMTask trainer."""

    tester: Tester = field(repr=False)
    """The configured LMTask tester used for held-out inference."""

    metrics_calculator: MetricsCalculator = field()
    """The task-specific metric implementation selected by the ``trainconf``
    configuration.

    """
    executor: Executor = field()
    """The command executor used to collect external provenance such as Git and
    NVIDIA information.

    """
    result_dir: Path = field()
    """Output director for the generated benchmark files."""

    template_dir: Path = field()
    """The directory containing benchmark Jinja2 templates."""

    temporary_dir: Path = field()
    """Directory to story temporary files."""

    detail_template: str = field(default='overview.md.jinja2')
    """The Jinja2 template filename used to render the per-benchmark Markdown
    report.

    """
    prediction_format: str = field(default='jsonl')
    """The persisted prediction format; reserved for future formats beyond
    JSONL.

    """
    class_count_column: str | None = field(default=None)
    """The dataset column used for per-split class counts; ``None`` disables
    class counting.

    """
    def __post_init__(self):
        self._result = PersistedWork(
            path=self.temporary_dir / 'result.pkl',
            owner=self,
            mkdir=True)

    def _exec(self, command: str) -> str:
        """Execute ``command`` and return its standard output."""
        out = StringIO()
        err = StringIO()
        executor: Executor = replace(self.executor, out=out, err=err)
        executor(command)
        err_str: str = err.getvalue()
        if len(err_str) > 0:
            logger.warning(f'{command}: {err_str}')
        return out.getvalue().strip()

    def _git_result(self) -> GitResult:
        """Return the current LMTask Git revision and working-tree state."""
        commit = self._exec('git rev-parse HEAD')
        describe = self._exec('git describe --tags --always --dirty')
        dirty = len(self._exec('git status --porcelain')) > 0
        return GitResult(commit=commit, describe=describe, dirty=dirty)

    def _package_versions(self) -> dict[str, str]:
        """Return exact versions of benchmark-relevant Python packages."""
        versions = {}
        for name in self._PACKAGE_NAMES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        return versions

    def _nvidia_driver(self) -> str | None:
        """Return the NVIDIA driver version when ``nvidia-smi`` is available."""
        try:
            return self._exec(
                "nvidia-smi --query-gpu=driver_version "
                "--format=csv,noheader | head -1")
        except OSError:
            return None

    def _environment(self, cuda: CudaInfo) -> EnvironmentResult:
        """Collect software and CUDA environment metadata."""
        gpus = tuple(
            GpuResult(
                index=index,
                name=device['name'],
                total_memory=device['memory']['total'])
            for index, device in cuda.get_devices().items())
        try:
            import torch
            cuda_runtime = torch.version.cuda
        except Exception:
            cuda_runtime = None
        return EnvironmentResult(
            python=platform.python_version(),
            os=platform.platform(),
            kernel=platform.release(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_runtime=cuda_runtime,
            nvidia_driver=self._nvidia_driver(),
            packages=self._package_versions(),
            gpus=gpus)

    def _split(self, name: str, source) -> DatasetSplit:
        """Collect size and optional class counts for one dataset source."""
        ds = source.create()
        counts = None
        if self.class_count_column is not None and \
           self.class_count_column in ds.column_names:
            counts = dict(Counter(map(
                str, ds[self.class_count_column])))
        return DatasetSplit(
            name=name,
            examples=len(ds),
            class_counts=counts)

    @property
    def datasets(self) -> DatasetSplit:
        """Collect metadata for configured train, validation and test splits."""
        splits = []
        if self.trainer.train_source is not None:
            splits.append(self._split('train', self.trainer.train_source))
        if self.trainer.eval_source is not None:
            splits.append(self._split(
                'validation', self.trainer.eval_source))
        if self.tester.source is not None:
            splits.append(self._split('test', self.tester.source))
        return DatasetSplitsResult(splits)

    @staticmethod
    def _directory_size(path: Path) -> int | None:
        """Return the persisted PEFT adapter payload size in bytes."""
        adapter = path / 'adapter_model.safetensors'
        return adapter.stat().st_size if adapter.is_file() else None

    def _write_json(self, result: BenchmarkResult) -> Path:
        """Persist the machine-readable benchmark source record."""
        path = self.result_dir / f'{self.name}.json'
        with open(path, 'w') as f:
            json.dump(result.asdict(), f, indent=4)
        logger.info(f'wrote: {path}')
        return path

    def _write_markdown(self, result: BenchmarkResult) -> Path:
        """Render the human-readable benchmark Markdown report."""
        env = Environment(
            loader=FileSystemLoader(self.template_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True)
        template = env.get_template(self.detail_template)
        path = self.result_dir / f'{self.name}.md'
        path.write_text(template.render(benchmark=result))
        logger.info(f'wrote: {path}')
        return path

    def _test(self) -> TestResult:
        result: TestResult
        if self.tester.result_exists:
            result = self.tester.load_result()
        else:
            result = self.tester.test()
            self.tester.save_result(result)
        return result

    @property
    @persisted('_result')
    def result(self) -> BenchmarkResult:
        """Train, test, score, persist and render this benchmark."""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        cuda = CudaInfo()
        environment: EnvironmentResult = self._environment(cuda)
        datasets: DatasetSplitsResult = self.datasets
        train_result: TrainResult

        if hasattr(cuda, 'reset_peak_memory_stats'):
            cuda.reset_peak_memory_stats()

        if not self.trainer.model_exists:
            train_result = self.trainer.train()
            self.trainer.save_result(train_result)
        else:
            train_result = self.trainer.load_result()

        adapter_size = self._directory_size(train_result.peft_output_dir)
        test_result: TestResult = self._test()
        df: pd.DataFrame = test_result.dataframe
        metrics: MetricsResult = self.metrics_calculator.calculate(df)
        pred_file: Path = self.result_dir / f'{self.name}.jsonl'

        if hasattr(cuda, 'get_peak_memory'):
            peaks = cuda.get_peak_memory()
            environment.gpus = tuple(
                replace(
                    gpu,
                    peak_allocated=peaks[gpu.index]['allocated'],
                    peak_reserved=peaks[gpu.index]['reserved'])
                for gpu in environment.gpus)

        return BenchmarkResult(
            name=self.name,
            task_name=self.task_name,
            model_name=self.model_name,
            config_file=str(self.config_file),
            created=datetime.now().astimezone().isoformat(),
            git=self._git_result(),
            environment=environment,
            datasets=datasets,
            training=TrainingResult(
                elapsed_seconds=train_result.time_elapsed,
                global_step=train_result.global_step,
                training_loss=train_result.training_loss,
                metrics=train_result.metrics,
                result_dir=str(train_result.peft_output_dir),
                adapter_size=adapter_size),
            testing=TestingResult(
                test_result=test_result,
                predictions_file=pred_file),
            metrics=metrics)

    def save_benchmark(self) -> BenchmarkResult:
        """Write the benchmark files."""
        result: BenchmarkResult = self.result
        result.testing.test_result.write_jsonl(result.testing.predictions_file)
        self._write_json(result)
        self._write_markdown(result)
        return result

    def clear(self):
        """Remove all cached data."""
        self._result.clear()
