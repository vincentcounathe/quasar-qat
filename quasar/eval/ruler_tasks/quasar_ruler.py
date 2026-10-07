"""lm-eval RULER builders with a disk cache and one sizing tokenizer per model family.

Loaded by lm-eval from the yamls in this directory (``--include_path``). Upstream
(``lm_eval/tasks/ruler``) synthesizes 500 samples per task and length by repeatedly
tokenizing haystacks with the evaluated model's tokenizer. Here:

1. every sample is sized with ``QUASAR_RULER_TOKENIZER`` (the family teacher), so all
   checkpoints of a family see identical contexts;
2. each (task, length) is cached under ``QUASAR_RULER_DATA``, keyed by that tokenizer and
   built with seeded RNGs, so it is a deterministic function of (task, length, tokenizer);
3. with ``QUASAR_RULER_EMPTY_THINK=1`` (thinking models) the QA answer prefix starts
   with an empty thinking block, so the short answer budget is not spent on reasoning.
"""

import hashlib
import os
import random
import shutil
import sys

import datasets
import numpy as np
from lm_eval.tasks.ruler import common_utils, niah_utils, qa_utils, vt_utils

from quasar.eval.settings import RULER_TASKS, SEED

TOKENIZER = os.environ["QUASAR_RULER_TOKENIZER"]
DATA = os.environ["QUASAR_RULER_DATA"]
EMPTY_THINK = os.environ["QUASAR_RULER_EMPTY_THINK"] == "1"
EMPTY_THINK_PREFIX = "<think>\n\n</think>\n\n"

process_results = common_utils.process_results
process_results_part = common_utils.process_results_part
aggregate_metrics = common_utils.aggregate_metrics


def _cached(name, builder, qa=False):
    def build(**kwargs):
        lengths = kwargs["max_seq_lengths"]  # one length per run (ruler.pregen / --metadata)
        key = hashlib.sha1(TOKENIZER.encode()).hexdigest()[:8]
        path = os.path.join(DATA, f"{name}_L{'_'.join(map(str, lengths))}_{key}")
        if not os.path.exists(os.path.join(path, "_DONE")):
            # The upstream builders draw from the global RNGs: seed them so a cache is a
            # function of (task, length, tokenizer) alone.
            random.seed(SEED)
            np.random.seed(SEED)
            tmp = f"{path}.tmp{os.getpid()}"
            builder(**dict(kwargs, tokenizer=TOKENIZER))["test"].save_to_disk(tmp)  # "tokenizer" wins over "pretrained"
            shutil.rmtree(path, ignore_errors=True)
            os.rename(tmp, path)
            open(os.path.join(path, "_DONE"), "w").close()
        ds = datasets.load_from_disk(path)
        if qa and EMPTY_THINK:
            ds = ds.map(lambda ex: {"gen_prefix": EMPTY_THINK_PREFIX + ex["gen_prefix"]})
        return {"test": ds}

    return build


niah_multikey_1 = _cached("niah_multikey_1", niah_utils.niah_multikey_1)
niah_multikey_2 = _cached("niah_multikey_2", niah_utils.niah_multikey_2)
niah_multikey_3 = _cached("niah_multikey_3", niah_utils.niah_multikey_3)
get_vt_dataset = _cached("ruler_vt", vt_utils.get_vt_dataset)
get_squad = _cached("ruler_qa_squad", qa_utils.get_squad, qa=True)
get_hotpotqa = _cached("ruler_qa_hotpot", qa_utils.get_hotpotqa, qa=True)
BUILDERS = {
    "niah_multikey_1": niah_multikey_1,
    "niah_multikey_2": niah_multikey_2,
    "niah_multikey_3": niah_multikey_3,
    "ruler_vt": get_vt_dataset,
    "ruler_qa_squad": get_squad,
    "ruler_qa_hotpot": get_hotpotqa,
}

if __name__ == "__main__":  # quasar.eval.ruler.pregen: build every cache at the given lengths
    for length in map(int, sys.argv[1:]):
        for task in RULER_TASKS:
            BUILDERS[task](max_seq_lengths=[length])
