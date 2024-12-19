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
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


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


# [ { "from": 35, "label": "general", "polarity": "positive", "text": "需要", "to": 37 }, { "from": 60, "label": "general", "polarity": "negative", "text": "着工ペース", "to": 65 }, { "from": 69, "label": "sales", "polarity": "positive", "text": "売上", "to": 71 } ]
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
    instruction_template: str = MT_INSTRUCTION_TEMPLATE,
):
    original_colunm_name = original_colunm_name or f"{original_lang}"
    target_colunm_name = target_colunm_name or f"{target_lang}"
    instruction = instruction_template.format(
        original_lang=original_lang,
        target_lang=target_lang,
    )

    def mt_prompt_fn(line, task_name: str = None):
        original_text = line[original_colunm_name]
        if len(original_text) > 2048:
            original_text = original_text[:2048]

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


task_alt_e_to_j = LightevalTaskConfig(
    name="ja:alt_e_to_j",
    prompt_function=build_mt_prompt_fn("english", "japanese"),
    suite=["community"],
    hf_repo="zenless-lab/alt",
    hf_subset="default",
    hf_avail_splits=["test", "train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select=None,
    generation_size=1024,
    stop_sequence=["\n"],
    metric=[
        Metrics.f1_score,
        # Metrics.bleu,
        # Metrics.bert_score,  # TODO: Add chunked bert_score metric
    ],
)


TASKS_TABLE = [
    jamp_task, janli_task, jsick_task, jsick_stress_task, jsem_task,
    jnli_task, jsquad_task, jcommonsenseqa_task, kuci_task,
    jcommonsense_morality_task, chabsa_extract_task, chabsa_polarity_task,
    mawps_task, mgsm_task, task_alt_e_to_j,
]
