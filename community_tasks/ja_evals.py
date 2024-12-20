# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr

from lighteval.metrics.metrics import CorpusLevelMetric, Metrics, SampleLevelMetricGrouping
from lighteval.metrics.metrics_corpus import (
    CorpusLevelTranslationMetric,
)
from lighteval.metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    GenerativePreparator,
)
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# NOTE: When performing translation tasks, the janome library needs to be installed.
#       Since Japanese does not use spaces for tokenization, the janome library is required for "wakati-gaki".
#       Please install the janome library using `pip install janome`.
try:
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer(wakati=True)
except ImportError:
    tokenizer = None


class CorpusLevelJapaneseTranslationMetric(CorpusLevelTranslationMetric):
    def __init__(self, metric_name: str):
        super().__init__(metric_name)

    def compute(self, items: list[GenerativeCorpusMetricInput]) -> float:
        if not tokenizer:
            raise ImportError(
                "The janome library is required to use the CorpusLevelJapaneseTranslationMetric class."
                "Please install it via `pip install janome`."
            )

        def wakati(item):
            item.golds = [" ".join(tokenizer.tokenize(gold)) for gold in item.golds]
            item.preds = [" ".join(tokenizer.tokenize(pred)) for pred in item.preds]
            return item

        items = [wakati(item) for item in items]
        return super().compute(items)


# Metrics
ja_bleu = CorpusLevelMetric(
    metric_name="bleu",
    sample_level_fn=GenerativePreparator().prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelJapaneseTranslationMetric("bleu").compute,
    higher_is_better=True,
)
ja_chrf = CorpusLevelMetric(
    metric_name="chrf",
    sample_level_fn=GenerativePreparator().prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelJapaneseTranslationMetric("chrf").compute,
    higher_is_better=True,
)
ja_ter = CorpusLevelMetric(
    metric_name="ter",
    sample_level_fn=GenerativePreparator().prepare,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=CorpusLevelJapaneseTranslationMetric("ter").compute,
    higher_is_better=False,
)


# Metrics
def correlation_metric(golds: list[int], predictions: list[str], **kwargs):
    def convert_to_float(score):
        try:
            return float(score)
        except ValueError:
            return None

    predicted_score = convert_to_float(predictions[0])
    gold_score = convert_to_float(golds[0])

    return {
        "predicted_score": predicted_score,
        "gold_score": gold_score,
    }


def spearman_corpus_metric(items):
    predicted_scores, gold_scores = zip(
        *[
            (item["predicted_score"], item["gold_score"])
            for item in items
            if (item["gold_score"] is not None and item["predicted_score"] is not None)
        ]
    )
    r, _ = spearmanr(predicted_scores, gold_scores)
    if np.isnan(r):
        return 0.0
    frac = len(predicted_scores) / len(items)

    return r * frac


def pearson_corpus_metric(items):
    predicted_scores, gold_scores = zip(
        *[
            (item["predicted_score"], item["gold_score"])
            for item in items
            if (item["gold_score"] is not None and item["predicted_score"] is not None)
        ]
    )
    r, _ = pearsonr(predicted_scores, gold_scores)
    if np.isnan(r):
        return 0.0
    frac = len(predicted_scores) / len(items)
    return r * frac


spearman_metric = CorpusLevelMetric(
    metric_name="spearman_correlation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.NONE,
    sample_level_fn=correlation_metric,
    corpus_level_fn=spearman_corpus_metric,
)

pearson_metric = CorpusLevelMetric(
    metric_name="pearson_correlation",
    higher_is_better=True,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.NONE,
    sample_level_fn=correlation_metric,
    corpus_level_fn=pearson_corpus_metric,
)


# NLI

NLI_INSTRUCTION = """\
前提と仮説の関係を「含意」、「矛盾」、「中立」の中から回答してください。
制約:
- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合は 含意 と出力
- 前提と仮説が両立しえない場合は 矛盾 と出力
- そのいずれでもない場合は 中立 と出力"""

NLI_PROMPT_TEMPLAT = """\
[前提]: {premise}
[仮説]: {hypothesis}
[関係]: """

NLI_LABELS = ["含意", "中立", "矛盾"]


def nli_prompt_fn(line, task_name: str = None):
    prompt = NLI_PROMPT_TEMPLAT.format(
        premise=line["premise"],
        hypothesis=line["hypothesis"]
    )
    query = NLI_INSTRUCTION + "\n\n" + prompt
    label = line["label"]
    if label is None:
        label = 1
        import logging
        logging.warning(f"Missing label for NLI task: {line}")

    return Doc(
        task_name=task_name,
        query=query,
        choices=NLI_LABELS,
        gold_index=label,
        instruction=NLI_INSTRUCTION,
    )


# JAMP

jamp_task = LightevalTaskConfig(
    name="ja:jamp",
    prompt_function=nli_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jamp",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# JANLI

JANLI_INSTRUCTION = """
前提と仮説の関係を「含意」、「含意ではない」の中から回答してください。
制約:
- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合は 含意 と出力
- 前提と仮説が両立しえない場合或いは含意関係にない場合は 含意ではない と出力"""

JANLI_LABELS = ["含意", "含意ではない"]


def janli_prompt_fn(line, task_name: str = None):
    prompt = NLI_PROMPT_TEMPLAT.format(
        premise=line["premise"],
        hypothesis=line["hypothesis"]
    )
    query = JANLI_INSTRUCTION + "\n\n" + prompt
    label = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=JANLI_LABELS,
        gold_index=label,
        instruction=JANLI_INSTRUCTION,
    )


janli_task = LightevalTaskConfig(
    name="ja:janli",
    prompt_function=janli_prompt_fn,
    suite=["community"],
    hf_repo="hpprc/janli",
    hf_subset="base",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# Jsick
jsick_task = LightevalTaskConfig(
    name="ja:jsick",
    prompt_function=nli_prompt_fn,
    suite=["community"],
    hf_repo="hpprc/jsick",
    hf_subset="base",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

jsick_stress_task = LightevalTaskConfig(
    name="ja:jsick_stress",
    prompt_function=nli_prompt_fn,
    suite=["community"],
    hf_repo="hpprc/jsick",
    hf_subset="stress",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)


# JSem
jsem_task = LightevalTaskConfig(
    name="ja:jsem",
    prompt_function=nli_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jsem",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# JNLI
jnli_task = LightevalTaskConfig(
    name="ja:jnli",
    prompt_function=nli_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jnli",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)


# Reading Comprehension

# JSQUAD
# JSQUAD_INSTRUCTION = "[題名]と[問題]から[質問]に対する[答え]を抜き出しなさい"  # The original prompt words in the paper
JSQUAD_INSTRUCTION = (
    "[質問]に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。 それ以外には何も含めないことを厳守してください。"
)

JSQUAD_PROMPT_TEMPLAT = """\
[題名]: {title}
[問題]: {context}
[質問]: {question}
[答え]: """


def jsquad_prompt_fn(line, task_name: str = None):
    prompt = JSQUAD_PROMPT_TEMPLAT.format(
        title=line["title"],
        context=line["context"],
        question=line["question"]
    )
    query = JSQUAD_INSTRUCTION + "\n\n" + prompt
    answer = line["answers"][0]["text"]

    doc = Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=JSQUAD_INSTRUCTION,
    )
    return doc


jsquad_task = LightevalTaskConfig(
    name="ja:jsquad",
    prompt_function=jsquad_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jsquad",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)


# Multiple Choice question answering

# MC_INSTRUCTION = "[問題]に対する[答え]を[選択肢]の中から選んでください。 "  # The original prompt words in the paper
MC_INSTRUCTION = (
    "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。"
    "なお、回答は選択肢の内容をそのまま出力し、他には何も含めないことを厳守してください。"
)

MC_INSTRUCTION_PROMPT_TEMPLAT = """\
[問題]: {question}
[選択肢]: {choices}
[答え]: """


def build_mc_choices_fn(
    choice_colunm_names: list[str],
    question_colunm_name: str = "question",
    instruction: str = MC_INSTRUCTION,
):
    def mc_prompt_fn(line, task_name: str = None):
        choices = [line[choice_colunm_name] for choice_colunm_name in choice_colunm_names]
        prompt = MC_INSTRUCTION_PROMPT_TEMPLAT.format(
            question=line[question_colunm_name],
            choices=str(choices)
        )
        query = instruction + "\n\n" + prompt
        label = line["label"]

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=label,
            instruction=instruction,
        )

    return mc_prompt_fn

# JCommonsenseQA


jcommonsenseqa_task = LightevalTaskConfig(
    name="ja:jcommonsenseqa",
    prompt_function=build_mc_choices_fn([f"choice{i}" for i in range(5)]),
    suite=["community"],
    hf_repo="zenless-lab/jcommonsenseqa",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# KUCI
kuci_task = LightevalTaskConfig(
    name="ja:kuci",
    prompt_function=build_mc_choices_fn(
        [f"choice{i}" for i in range(4)],
        question_colunm_name="context",
        instruction=(
            "文脈と選択肢を入力として受け取り、選択肢から文脈の後に続く文として最も適切なものを選択してください。"
            "なお、回答は選択肢の内容をそのまま出力し、他には何も含めないことを厳守してください。"
        )
    ),
    suite=["community"],
    hf_repo="zenless-lab/kuci",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# JCommonsenseMorality
JCOMMONSENSE_MORALITY_INSTRUCTION = (
    "これから提示する日本語の文章が道徳的な誤りを含んでいるかを判断してください。"
    "道徳的に正しい文章の場合「道徳的」、誤りを含んでいる場合は「不道徳的」を出力してください。"
    "必ず 道徳的 か 不道徳的 のどちらかを出力し、それ以外には何も含めないことを厳守してください。"
)

JCOMMONSENSE_MORALITY_PROMPT_TEMPLAT = """\
[文章]: {context}
[判断]: """


def jcommonsense_morality_prompt_fn(line, task_name: str = None):
    prompt = JCOMMONSENSE_MORALITY_PROMPT_TEMPLAT.format(
        context=line["question"]
    )
    query = JCOMMONSENSE_MORALITY_INSTRUCTION + "\n\n" + prompt
    label = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=["道徳的", "不道徳的"],
        gold_index=label,
        instruction=JCOMMONSENSE_MORALITY_INSTRUCTION,
    )


jcommonsense_morality_task = LightevalTaskConfig(
    name="ja:jcommonsense_morality",
    prompt_function=jcommonsense_morality_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jcommonsensemorality",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)


# Entity Linking
# chABSA
CHABSA_EXTRACT_INSTRUCTION = (
    "与えられた文章から固有表現で書かれたターゲットの名前を抽出して下さい。"
    "固有表現で書かれたターゲットの名前を半角コンマ（,）で区切って出力し、それ以外には何も含めないことを厳守してください。"
    "ただし、ターゲットは固有表現である市場、市況、会社/法人、グループ、会社内の部門、事業部、事業領域、"
    "製品、サービスの名称などを指すこととします。"
)

CHABSA_EXTRACT_PROMPT_TEMPLAT = """\
[文章]: {context}
[ターゲット]: """


def chabsa_extract_prompt_fn(line, task_name: str = None):
    prompt = CHABSA_EXTRACT_PROMPT_TEMPLAT.format(
        context=line["sentence"]
    )
    query = CHABSA_EXTRACT_INSTRUCTION + "\n\n" + prompt
    answer = ", ".join([entity["text"] for entity in line["opinions"]])

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=CHABSA_EXTRACT_INSTRUCTION,
    )


chabsa_extract_task = LightevalTaskConfig(
    name="ja:chabsa_extract",
    prompt_function=chabsa_extract_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/chABSA",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.f1_score,
        Metrics.f1_score_quasi,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)

CHABSA_POLARITY_INSTRUCTION = (
    "与えられた文章から固有表現に対する極性をpositive、neutral、negativeの中から選択して下さい。"
    "固有表現の極性（positive、neutral、negativeのいずれか）のペアを半角コンマ（,）で区切って出力し、"
    "それ以外には何も含めないことを厳守してください。"
)

CHABSA_POLARITY_PROMPT_TEMPLAT = """\
[文章]: {context}
[固有表現]: {entity}
[極性]: """


def chabsa_polarity_prompt_fn(line, task_name: str = None):
    entity = ",".join([f"{entity['text']}, {entity['polarity']}" for entity in line["opinions"]])
    prompt = CHABSA_POLARITY_PROMPT_TEMPLAT.format(
        context=line["sentence"],
        entity=entity
    )
    query = CHABSA_POLARITY_INSTRUCTION + "\n\n" + prompt
    answer = ",".join([f"{entity['text']}, {entity['polarity']}" for entity in line["opinions"]])

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=CHABSA_POLARITY_INSTRUCTION,
    )


chabsa_polarity_task = LightevalTaskConfig(
    name="ja:chabsa_polarity",
    prompt_function=chabsa_polarity_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/chABSA",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.f1_score,
        Metrics.f1_score_quasi,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)

# Fundamental Analysis
# TODO: Wikipedia Annotated Corpus

# Mathematical Reasoning
MR_INSTRUCTION = (
    "与えられた計算問題に対する答えを整数または小数で出力してください。"
    "数値のみを出力し、それ以外には何も含めないことを厳守してください。"
)

MR_PROMPT_TEMPLAT = """\
[問題]: {question}
[答え]: """


def build_mr_prompt_fn(
    question_colunm_name: str = "question",
    answer_colunm_name: str = "answer",
    instruction: str = MR_INSTRUCTION,
    is_float_colunm_name: str | None = None,
):
    def mr_prompt_fn(line, task_name: str = None):
        prompt = MR_PROMPT_TEMPLAT.format(
            question=line[question_colunm_name]
        )
        query = instruction + "\n\n" + prompt
        if is_float_colunm_name and line[is_float_colunm_name]:
            answer = str(float(line[answer_colunm_name]))
        else:
            answer = str(int(line[answer_colunm_name]))

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=instruction,
        )

    return mr_prompt_fn


# MAWPS
mawps_task = LightevalTaskConfig(
    name="ja:mawps",
    prompt_function=build_mr_prompt_fn(is_float_colunm_name="is_float"),
    suite=["community"],
    hf_repo="zenless-lab/mawps",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.quasi_exact_match_gsm8k,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)

# MGSM
mgsm_task = LightevalTaskConfig(
    name="ja:mgsm",
    prompt_function=build_mr_prompt_fn(answer_colunm_name="answer_number"),
    suite=["community"],
    hf_repo="juletxara/mgsm",
    hf_subset="ja",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.f1_score_macro,
        Metrics.f1_score_micro,
    ],
)

# Machine Translation
MT_INSTRUCTION_TEMPLATE = (
    "これから提示する{original_lang}の文章を{target_lang}に翻訳してください。"
    "必ず{target_lang}の訳文のみを出力し、それ以外には何も含めないことを厳守してください。"
)

MT_PROMPT_TEMPLATE = """\
[原文]: {original}
[訳文]: """


def build_mt_prompt_fn(
    original_lang: str,
    target_lang: str,
    original_colunm_name: str | None = None,
    target_colunm_name: str | None = None,
    cutoff_length: int = 1024,
    instruction_template: str = MT_INSTRUCTION_TEMPLATE,
):
    try:
        from janome.tokenizer import Tokenizer
    except ImportError:
        raise ImportError(
            "The janome library is required to use the build_mt_prompt_fn function."
            "Please install it via pip install janome."
        )

    original_colunm_name = original_colunm_name or f"{original_lang}"
    target_colunm_name = target_colunm_name or f"{target_lang}"
    instruction = instruction_template.format(
        original_lang=original_lang,
        target_lang=target_lang,
    )

    def mt_prompt_fn(line, task_name: str = None):
        original_text = line[original_colunm_name]
        if len(original_text) > cutoff_length:
            original_text = original_text[:cutoff_length]

        prompt = MT_PROMPT_TEMPLATE.format(
            original=line[original_colunm_name]
        )
        query = instruction + "\n\n" + prompt
        answer = line[target_colunm_name]

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=instruction,
        )

    return mt_prompt_fn


# ALT
def build_mt_task(
        name: str,
        hf_repo: str,
        original_lang: str,
        target_lang: str,
        subset: str,
        generation_size: int = 1024,
):
    lang_name_map = {
        "english": "英語",
        "japanese": "日本語",
    }

    return LightevalTaskConfig(
        name=name,
        prompt_function=build_mt_prompt_fn(
            lang_name_map[original_lang], lang_name_map[target_lang], original_lang, target_lang
            ),
        suite=["community"],
        hf_repo=hf_repo,
        hf_subset=subset,
        hf_avail_splits=["test", "train"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select=None,
        generation_size=generation_size,
        stop_sequence=["\n"],
        metric=[
            ja_bleu,
            ja_chrf,
            ja_ter,
        ],
    )


alt_tasks = [
    build_mt_task("ja:alt:e_to_j", "zenless-lab/alt", "english", "japanese", "default", 4096),
    *[
        build_mt_task(f"ja:alt_{key}:e_to_j", "zenless-lab/alt", "english", "japanese", f"{key}-token", value)
        for key, value in {
            "256": 256,
            "512": 512,
            "1k": 1024,
            "2k": 2048,
        }.items()
    ],
    build_mt_task("ja:alt:j_to_e", "zenless-lab/alt", "japanese", "english", "default", 4096),
    *[
        build_mt_task(f"ja:alt_{key}:j_to_e", "zenless-lab/alt", "japanese", "english", f"{key}-token", value)
        for key, value in {
            "256": 256,
            "512": 512,
            "1k": 1024,
            "2k": 2048,
        }.items()
    ],
]


# Semantic Text Similarity
# JSTS
JSTS_INSTRUCTION = (
    "日本語の文ペアの意味がどのくらい近いかを判定し、類似度を0.0〜5.0までの間の値で付与してください。"
    "0.0に近いほど文ペアの意味が異なり、5.0に近いほど文ペアの意味が似ていることを表しています。"
    "整数値のみを返し、それ以外には何も含めないことを厳守してください。"
)
JSTS_PROMPT_TEMPLAT = """\
[文1]: {sentence1}
[文2]: {sentence2}
[類似度]: """


def jsts_prompt_fn(line, task_name: str = None):
    prompt = JSTS_PROMPT_TEMPLAT.format(
        sentence1=line["sentence1"],
        sentence2=line["sentence2"]
    )
    query = JSTS_INSTRUCTION + "\n\n" + prompt
    answer = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=JSTS_INSTRUCTION,
    )


jsts_task = LightevalTaskConfig(
    name="jglue:jsts",
    prompt_function=jsts_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jsts",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[spearman_metric, pearson_metric],
)

wikicorpus_tasks = [
    build_mt_task("ja:wikicorpus_e_to_j", "zenless-lab/wikicorpus", "english", "japanese", "default", 4096),
    build_mt_task("ja:wikicorpus_j_to_e", "zenless-lab/wikicorpus", "japanese", "english", "default", 4096),
]


# Human Examination
HE_INSTRUCTION = (
    "与えられた質問と選択肢から、最も適切な回答を選択してください。"
    "なお、回答には選択肢の内容をそのまま出力し、他には何も含めないことを厳守してください。"
)

HE_PROMPT_TEMPLAT = """\
[質問]: {question}
[選択肢]: {choices}
[回答]: """


def jmmlu_prompt_fn(line, task_name: str = None):
    prompt = f"{line['question']}\n"
    query = HE_INSTRUCTION + "\n\n" + prompt
    choices = [line[f"choice{i}"] for i in range(4)]
    answer = line["label"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer,
        instruction=HE_INSTRUCTION,
    )

# JMMLU


jmmlu_task = LightevalTaskConfig(
    name="ja:jmmlu",
    prompt_function=jmmlu_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jmmlu",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

jmmlu_nc_nd_task = LightevalTaskConfig(
    name="ja:jmmlu_nc_nd",
    prompt_function=jmmlu_prompt_fn,
    suite=["community"],
    hf_repo="zenless-lab/jmmlu-nc-nd",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)


# MMMLU
def mmmlu_prompt_fn(line, task_name: str = None):
    answer_index_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    prompt = f"{line['Question']}\n"
    query = HE_INSTRUCTION + "\n\n" + prompt
    choices = [line[key] for key in answer_index_map.keys()]
    answer = answer_index_map[line["Answer"]]

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer,
        instruction=HE_INSTRUCTION,
    )


mmmlu_task = LightevalTaskConfig(
    name="ja:mmmlu",
    prompt_function=mmmlu_prompt_fn,
    suite=["community"],
    hf_repo="openai/MMMLU",
    hf_subset="JA_JP",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    stop_sequence=["\n"],
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.loglikelihood_acc,
        Metrics.loglikelihood_acc_norm,
        Metrics.loglikelihood_acc_norm_nospace,
    ],
)

# Code Generation
# MBPP
# TODO: Code Generation tasks(llm-jp/mbpp-ja)

# Summarization
# XL-Sum
XL_SUM_INSTRUCTION = (
    "与えられたニュース記事を要約してください。"
)
XL_SUM_PROMPT_TEMPLAT = """\
[記事]: {article}
[要約]: """


def xl_sum_prompt_fn(line, task_name: str = None):
    prompt = XL_SUM_PROMPT_TEMPLAT.format(
        article=line["text"]
    )
    query = XL_SUM_INSTRUCTION + "\n\n" + prompt
    answer = line["summary"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[answer],
        gold_index=0,
        instruction=XL_SUM_INSTRUCTION,
    )


xl_sum_task = LightevalTaskConfig(
    name="ja:xl_sum",
    prompt_function=xl_sum_prompt_fn,
    suite=["community"],
    hf_repo="csebuetnlp/xlsum",
    hf_subset="japanese",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=512,
    stop_sequence=["\n"],
    metric=[
        Metrics.rouge1,
        Metrics.rouge2,
        Metrics.rougeL,
        ja_bleu,
    ],
)


TASKS_TABLE = [
    jamp_task, janli_task, jsick_task, jsick_stress_task, jsem_task,
    jnli_task, jsquad_task, jcommonsenseqa_task, kuci_task,
    jcommonsense_morality_task, chabsa_extract_task, chabsa_polarity_task,
    mawps_task, mgsm_task, *alt_tasks, *wikicorpus_tasks,
    jsts_task, jmmlu_task, jmmlu_nc_nd_task, mmmlu_task,
    xl_sum_task,
]
