from pprint import pprint
import textwrap
import numpy as np
import datasets
from datasets import Dataset
from tqdm import tqdm
from zensols.lmtask.instruct import InstructTaskRequest
from zensols.lmtask import Task, TaskRequest, TaskResponse
from zensols.lmtask.torchconfig import TorchConfig
from util import TestBase


class TestInstruct(TestBase):
    DEBUG: bool = 0
    LIMIT: int = 50
    EXPECT_F1: int = 0.9
    TEST_DEFAULT: bool = False

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

    def _test_instruct(self, task_name: str, model: str, task: str = 'dataset',
                       clear: bool = False, min_f1: float = None) -> \
            dict[str, float]:
        name: str = f'{task_name}-{model}'
        task: Task = self._get_trained_task(task_name, model, task)
        if clear:
            task.generator.resource.clear()
        if task is None:
            return
        ds: Dataset = datasets.load_dataset('stanfordnlp/imdb', split='test')
        if self.DEBUG:
            print(f'testing: {task_name}-{model}')
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
                params = {}
                if task_name is not None:
                    params['include_model_output_raw'] = True
                res.write(**params)
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
        if min_f1 is None:
            self.assertTrue(
                mets['f1'] > self.EXPECT_F1,
                f'poor f1. (expect at least {self.EXPECT_F1}): {mets}')
        else:
            self.assertTrue(
                mets['f1'] < min_f1,
                f'high f1 for default model {min_f1}): {mets}')
        return mets

    def test_default_llama3(self):
        if self.TEST_DEFAULT:
            self._test_instruct(None, 'llama3', 'sentiment',
                                self.TEST_DEFAULT, 0.2)

    def test_imdb_llama3(self):
        self._test_instruct('imdb', 'llama3', 'dataset', self.TEST_DEFAULT)

    def test_imdb_qwen3(self):
        self._test_instruct('imdb', 'qwen3')

    def test_imdb_gemma4(self):
        #self._test_instruct(None, 'gemma4', 'sentiment', False)
        self._test_instruct('imdb', 'gemma4')

    def test_imdb_dsr1qwen(self):
        self._test_instruct('imdb', 'dsr1qwen3')
