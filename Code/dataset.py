'''
TextDataset class definition
Modified existing TextDatset class to circumvent pickle protocol error
Original code: https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py
'''


import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
from transformers import GPT2Tokenizer #added by me

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: GPT2Tokenizer, #changed by me
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.

        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
            )
        # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should look for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

        #start = time.time()
        #with open(cached_features_file, "wb") as handle:
        #    pickle.dump(self.examples, handle, protocol=3) #changed from HIGHEST_PROTOCOL


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)