# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch
import datetime
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .bert.bert_model import BertTripConfig

from util import dataset_metadata
import math
import pandas as pd
from sklearn.neighbors import BallTree

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_specfial_tokens_mask=True`.
    """
    dataset: str
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    config: BertTripConfig = None
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        self.user_start_pos = 1
        self.user_end_pos = self.user_start_pos + self.config.num_user_token
        self.time_start_pos = self.user_end_pos
        self.time_end_pos = self.time_start_pos + self.config.num_time_token
        self.poi_type_id = 1
        self.user_type_id = 2
        self.time_type_id = 3
        self.ignore_mask_id = -100
        
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        poi_examples = []
        aug_poi_examples = []

        for example in examples:
            poi_examples.append({"input_ids": example['input_ids']})
            aug_poi_examples.append({"input_ids": example['aug_input_ids']})
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['aug_input_ids'] = self.tokenizer.pad(aug_poi_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch['aug_attention_mask'] = batch['aug_input_ids']["attention_mask"]
        batch['aug_input_ids'] = batch['aug_input_ids']["input_ids"]

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"], batch["aug_input_ids"], batch["aug_labels"]= self.mask_tokens(
            batch["input_ids"], batch['aug_input_ids'], special_tokens_mask=special_tokens_mask
        )

        if not self.config.use_data_agumentation:
            if 'aug_input_ids' in batch:
                del batch['aug_input_ids']
            if 'aug_labels' in batch:
                del batch['aug_labels']
        return batch

    def data_agumentation(self, labels, special_tokens_mask, mln_probability = 0.15):
         if mln_probability == -1:
             mln_probability = torch.rand(1).uniform_(0.15, 0.5)[0]
         probability_matrix = torch.full(labels.shape, mln_probability)
         probability_matrix.masked_fill_(special_tokens_mask, value = 0.0)
         masked_indices = torch.bernoulli(probability_matrix).bool()
         return masked_indices

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        aug_inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        config = self.config
        # print(self.dataset)
        # print(dataset_metadata)
        USER_NUM = dataset_metadata[self.dataset]['USER_NUM']
        TIME_NUM = dataset_metadata[self.dataset]['TIME_NUM']
        # print("=====")
        # print(len(self.tokenizer), USER_NUM, TIME_NUM)
        TIME_START_INDEX = len(self.tokenizer) - USER_NUM - TIME_NUM

        USER_START_INDEX = TIME_START_INDEX + TIME_NUM

        labels = inputs.clone()
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        input_types = torch.full(labels.shape, self.poi_type_id)
        if config.add_user_token:
            input_types[:, self.user_start_pos:self.user_end_pos] = self.user_type_id
        if config.add_time_token:
            input_types[:, self.time_start_pos:self.time_end_pos] = self.time_type_id

        if config.use_data_agumentation:
            masked_indices = self.data_agumentation(labels, special_tokens_mask, -1)
        else:
            masked_indices = self.data_agumentation(labels, special_tokens_mask, 0.15)
        labels[~masked_indices] = self.ignore_mask_id

        #input
        is_poi = (input_types == self.poi_type_id)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices & is_poi
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & is_poi & ~indices_replaced
        # print(TIME_START_INDEX, labels.shape)
        # inputs_random_words = torch.randint(TIME_START_INDEX, labels.shape, dtype=torch.long)
        # inputs[indices_random] = inputs_random_words[indices_random]

        # user
        if config.add_user_token:
            is_user = (input_types == self.user_type_id)
            labels[is_user] = self.ignore_mask_id
            user_indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_user
            inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, labels.shape, dtype=torch.long)
            inputs[user_indices_random] = inputs_random_words[user_indices_random]

        # time
        if config.add_time_token:
            is_time = (input_types == self.time_type_id)
            labels[is_time] = self.ignore_mask_id
            indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & is_time
            inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, labels.shape, dtype=torch.long)
            inputs[indices_random] = inputs_random_words[indices_random]


        if config.use_data_agumentation:
            aug_inputs = aug_inputs.clone()
            aug_labels = aug_inputs.clone()
            aug_special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in aug_labels.tolist()]
            aug_special_tokens_mask = torch.tensor(aug_special_tokens_mask, dtype=torch.bool)
            aug_input_types = torch.full(aug_labels.shape, self.poi_type_id)
            if config.add_user_token:
                aug_input_types[:, self.user_start_pos:self.user_end_pos] = self.user_type_id
            if config.add_time_token:
                aug_input_types[:, self.time_start_pos:self.time_end_pos] = self.time_type_id

            aug_masked_indices = self.data_agumentation(aug_labels, aug_special_tokens_mask, -1)
            aug_labels[~aug_masked_indices] = self.ignore_mask_id

            #input
            aug_is_poi = (aug_input_types == self.poi_type_id)
            aug_indices_replaced = torch.bernoulli(torch.full(aug_labels.shape, 0.8)).bool() & aug_masked_indices & aug_is_poi
            aug_inputs[aug_indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 0.5)).bool() & aug_masked_indices & aug_is_poi & ~aug_indices_replaced
            aug_inputs_random_words = torch.randint(TIME_START_INDEX, aug_labels.shape, dtype=torch.long)
            aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]

            #user
            if config.add_user_token:
                aug_is_user = (aug_input_types == self.user_type_id)
                aug_labels[aug_is_user] = self.ignore_mask_id
                aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & aug_masked_indices & aug_is_user
                aug_inputs_random_words = torch.randint(USER_START_INDEX, USER_START_INDEX + USER_NUM, aug_labels.shape, dtype=torch.long)
                aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]

            #time
            if config.add_time_token:
                aug_is_time = (aug_input_types == self.time_type_id)
                aug_labels[aug_is_time] = self.ignore_mask_id
                aug_indices_random = torch.bernoulli(torch.full(aug_labels.shape, 1.0)).bool() & aug_masked_indices & aug_is_time
                aug_inputs_random_words = torch.randint(TIME_START_INDEX, TIME_START_INDEX + TIME_NUM, aug_labels.shape, dtype=torch.long)
                aug_inputs[aug_indices_random] = aug_inputs_random_words[aug_indices_random]
        else:
            aug_inputs = None
            aug_labels = None
        return inputs,  labels, aug_inputs, aug_labels
