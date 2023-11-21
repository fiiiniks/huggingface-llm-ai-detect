# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv

import datasets
from datasets.tasks import TextClassification

logger = datasets.logging.get_logger(__name__)


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "train": "https://fiiiniks.github.io/huggingface-llm-ai-detect/llm-detect-ai-generated-text/train_essays.csv",
    "test": "https://doc-0s-88-docs.googleusercontent.com/docs/securesc/puvkfhrtjrfkdvb9ute0p7a4qlfi2016/n5lc2e6tirjni34qva1pg8m93jspbkuv/1698973350000/17156827059415363851/17156827059415363851/1MF5nwV_OmItiMSY3qeJE48rpvJrmlR3s?e=download&ax=AI0foUpkg1WqY19770fGfY_m2GTndNVaWro6vBMbLXld0mZ6vn-8gmLm9l5tXwXOCuiKUKpZ9RI7evLb1vimKKlbGyYNSZTD6dPnwHhSZ5ufuKAxH6culOLnST9JTa1BoSDS2MrWNebBSukGbY9VMR0xDv8vBqDwfKD9uG6uRhcL4zTtUoqYKtKUk40SZ4qxTXA_hyP22JuVWGRyGbbpKkJRyeAPi-wbYiW6UxWO_5muUxSB6jg35gEUO28OXCg1ZI0aPWiV-YuSsE_hDbvsBKzVHyzO8-MnZZGOoPc5bGN6pTxKC8JNrSAE4I8DRAqA36AWTRvxM226p3Ko_hM-Cjkl-uHzparBg0BiqDY2BA_AixrAd9aU8JQ3LNie_i19VFa_I_mwnyWqDlleNlGqexAVU8twF-DKJC-TUq1EoA168tU6bj-omBih_TxdOOZXnEnNCH-f2vdA9ZMT-8nDXqdxbVcTvm8Fv5bcwm_CLqMrvHgqSEekuJPTBNmd5ZCYHwsGSIgGJsqJgnKZUupWtBHcFyAm7o3jAnrGcOPKDqZqxe1m8DpNT95r8jAXZp3GKfQ1MQZ_TFf0V8LCanNqhKpegbCmfTS0_YdjdiQWD9GB2vnX4VLxE5ctc5JM1jpNVL8b6KLNpskvoR62yRrm_VvJJPtyXbaj_oENHsJcshgNFuQSkRQVUwcob8NdAE1XTgE_iUIZFTgytuhFiLJN4ELpSHSZoMAsQ2pPHbvNCPVFH4Oflhj3uDNfP1O-sGONzYWij4pZdg8W6_N9pN0Iz-u0HQk_HnURGHRuKRAntYBc4gLlsPyG8cmyZkOkfCSzt8wmnlDE_OEQ834aO6D3kY5MBGOgx-SGo7SirfEL92KoBid76uzdCYacBO-6fm7Ek9tEaZImNhnZ37IPQcPlSAnOM2LauZufgdFBHOygm0K95O5yo_AsBD3C5e88V9zDHyEArnsZ4J_lfT8RfYB-Z7mJWsjjEAZ89hyLSSk5jqo2VDFLSg&uuid=80302254-a804-4224-b3be-7c2878a65cfa&authuser=0&nonce=j4qfmjomnaf9k&user=17156827059415363851&hash=3puoq1vjd5lrft7i7db6f6dfosfpo36p",
}


class AIDetectConfig(datasets.BuilderConfig):
    """BuilderConfig for AIDetect."""

    def __init__(self, **kwargs):
        """BuilderConfig forAIDetects.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(AIDetectConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class AIDetect(datasets.GeneratorBasedBuilder):
    """TODO: AI detect using LLM in writing essay."""

    BUILDER_CONFIGS = [
        AIDetectConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "prompt_id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "generated": datasets.features.ClassLabel(
                        num_classes=2
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="generated")
            ],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate AIDetect examples."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        # label_mapping = {"negative": 0, "positive": 1}
        with open(filepath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # label = label_mapping[row["label"]]
                yield key, {
                    "id": row["id"],
                    "prompt_id": row["prompt_id"],
                    "text": row["text"],
                    "generated": row["generated"],
                }
                key += 1
            