"""
A ``TokenEmbedder`` which uses one of the BERT models
(https://github.com/google-research/bert)
to produce embeddings.

At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-pretrained-BERT),
so thanks to them!
"""
from typing import Dict
import logging

import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel

from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import util
from allennlp_overrides.pytorch_pretrained_bert.modeling import LayeredBertModel

logger = logging.getLogger(__name__)



class LayeredPretrainedBertModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, BertModel] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> BertModel:
        if model_name in cls._cache:
            return LayeredPretrainedBertModel._cache[model_name]

        model = LayeredBertModel.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model

