from pprint import pprint
import warnings
import textwrap
import numpy as np
import datasets
from datasets import Dataset
from tqdm import tqdm
from zensols.lmtask import TaskFactory
from zensols.config import ConfigFactory
from zensols.lmtask.instruct import InstructTaskRequest
from zensols.lmtask import Task, TaskRequest, TaskResponse
from zensols.lmtask.torchconfig import TorchConfig
from util import TestBase


class TestImdbTrained(TestBase):
    DEBUG: bool = False
    LIMIT: int = 50
    EXPECT_F1: int = 0.9

    def setUp(self):
        super().setUp()
        TorchConfig.set_random_seed()

    def _get_binary_metrics(self, labels, preds, positive=1) -> \
            dict[str, float]:
        y = np.asarray(labels)
        p = np.asarray(preds)
        assert y.shape == p.shape
        tp = np.sum((y == positive) & (p == positive))
        tn = np.sum((y != positive) & (p != positive))
        fp = np.sum((y != positive) & (p == positive))
        fn = np.sum((y == positive) & (p != positive))
        accuracy = (tp + tn) / len(y) if len(y) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) \
            if (precision + recall) else 0.0
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)}

    def _test_imdb(self, model: str) -> dict[str, float]:
        task_name: str = 'imdb'
        if not self._trained_model_exists(task_name, model):
            warnings.warn(
                f"Trained model '{task_name}-{model}' does not exist--skipping",
                UserWarning)
            return
        fac: ConfigFactory = self._get_config_factory(task_name, model)
        name: str = f'{task_name}-{model}'
        task_factory: TaskFactory = fac('lmtask_task_factory')
        task: Task = task_factory.create('dataset')
        ds: Dataset = datasets.load_dataset('stanfordnlp/imdb', split='test')
        labels: list[str] = []
        preds: list[str] = []
        ds = ds.shuffle(seed=0)
        ds = ds.select(range(self.LIMIT))
        for i, review in tqdm(enumerate(ds), total=len(ds), desc=name):
            if self.DEBUG:
                print(('_' * 30), f'<{i}>', ('_' * 30))
            req: TaskRequest = InstructTaskRequest(instruction=review['text'])
            if self.DEBUG:
                task.prepare_request(req).write(include_instruction=False)
            res: TaskResponse = task.process(req)
            if self.DEBUG:
                res.write(include_model_output_raw=True)
            label: str = 'positive' if review['label'] == 1 else 'negative'
            pred: str = res.model_output.strip().lower()
            labels.append(label)
            preds.append(pred)
            correct: bool = (label == pred)
            pred_str: str = textwrap.shorten(pred, width=20)
            if self.DEBUG:
                print(f'correct: {correct} (label={label}, pred=<{pred_str}>)')
        mets: dict[str, float] = self._get_binary_metrics(
            labels, preds, positive='positive')
        if self.DEBUG:
            pprint(mets)
        self.assertTrue(
            mets['f1'] > self.EXPECT_F1,
            f'poor performance (expect at lest{self.EXPECT_F1}): {mets}')
        return mets

    def test_imdb_llama(self):
        model: str
        for model in 'llama3 qwen3 gemma4'.split():
            self._test_imdb(model)
