"""Paper evaluations. The CLIs (``python -m quasar.eval.<module> --help``):

  suite      one checkpoint on a whole table (healing, adaptation or nvfp4), task by task
  heldout    held-out perplexity, plus KL / top-1 agreement against the BF16 reference
  run        one generation task (math, MMLU-Pro, SuperGPQA, LiveCodeBench, LongBench-v2)
  lcb        LiveCodeBench scoring with the official execution-based evaluator
  ruler      RULER through lm-eval (+ CPU data pre-generation)

All table settings live in ``quasar.eval.settings``. vLLM, lm-eval, math-verify and
LiveCodeBench are optional dependencies, imported only by the tasks that need them.
"""
