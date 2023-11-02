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
    "train": "https://tidrael.github.io/tsl-news/train.csv",
    "test": "https://tidrael.github.io/tsl-news/test.csv",
}


class LLMDetectConfig(datasets.BuilderConfig):
    """BuilderConfig for LLM AI generator detect training."""

    def __init__(self, **kwargs):
        """BuilderConfig for LLMDetect.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LLMDetectConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class LLMDetect(datasets.GeneratorBasedBuilder):
    """TODO: Essay generate with AI and the self-writing."""

    BUILDER_CONFIGS = [
        LLMDetectConfig(
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
                    "promt_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "generated": datasets.features.ClassLabel(
                        names=[0, 1]
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="generate")
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
        """Generate LLMDetect examples."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        # label_mapping = {"negative": 0, "positive": 1}
        with open(filepath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # label = label_mapping[row["label"]]
                label = row["generated"]
                yield key, {
                    "id": row["id"],
                    "prompt_id": row["prompt_id"],
                    "text": row["text"],
                    "label": label,
                }
                key += 1