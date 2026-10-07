"""Chat rendering shared by training and evaluation (``chat``) and the corpus tools.

Corpus tools, each a CLI (``python -m quasar.data.<tool> --help``):
  prompts         seeded Open-PerfectBlend prompt slices, with exclusions.
  sample_teacher  teacher responses from an OpenAI-compatible server.
  build_distill   filter, dedup, length-check and split teacher samples.
  build_math      OpenMathReasoning adaptation corpus, decontaminated.

A distillation corpus is ``prompts`` -> ``sample_teacher`` -> ``build_distill``; each
module's docstring gives the command used for each paper corpus.
"""

# Shared by the adaptation corpus and the math evaluation, so training and evaluation
# prompts cannot drift apart: the math prompt suffix, and the reasoning prefix the
# corpus's assistant turns start with and thinking-mode evaluation seeds.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
THINK_OPEN = "<think>\n"
