# Agent-tool integration

LMTask specializes and serves a model-backed task. An orchestration framework
can then register that task as one of its tools.

The example remains framework-neutral so LMTask does not require LangChain,
LangGraph, or another agent runtime:

```python
from sentiment_tool import classify_sentiment

result = classify_sentiment(
    'The documentation was clear, but model startup was slow.')
print(result)
```

The callable can be registered with any framework that accepts Python
functions or JSON-compatible tool results. Keep planning, state, and tool
selection in the orchestration layer; keep model loading, task prompts,
response parsing, caching, and specialization in LMTask.
