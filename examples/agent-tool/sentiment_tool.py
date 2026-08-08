"""Expose a configured LMTask sentiment task as an application/agent tool."""
from typing import Any
from zensols.lmtask import ApplicationFactory, InstructTaskRequest


_factory = ApplicationFactory.get_task_factory()
_sentiment_task = _factory.create('sentiment')


def classify_sentiment(text: str) -> Any:
    """Classify one or more statements and return JSON-compatible output."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError('text must be a non-empty string')
    response = _sentiment_task.process(
        InstructTaskRequest(instruction=text))
    return response.model_output_json
