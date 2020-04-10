from typing import Dict, Union

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from allennlp_overrides.modules.token_embedders.layered_bert_token_embedder import LayeredPretrainedBertModel
from allennlp_overrides.training.metrics.count_n_layers import CountNLayers


@Model.register("multiloss_bert")
class MultilossBert(Model):
    """
    Train a BERT model, which makes predictions based on multiple layers.

    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    scaling_temperature: ``str``, optional (default: "1")
        Scaling temperature parameter of each layer for better calibration
    layer_indices: ``str``, optional (default: "23")
        Indices for layers for which linear layers are learned
    multitask: ``bool``, optional (default: false)
        Do multitask learning (rather than summing all losses)
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, LayeredPretrainedBertModel],
                 dropout: float = 0.0,
                 num_labels: int = None,
                 index: str = "bert",
                 label_namespace: str = "labels",
                 trainable: bool = True,
                 scaling_temperature: str = "1",
                 temperature_threshold: float = -1,
                 layer_indices: str = "23",
                 multitask: bool = False,
                 debug: bool = False,
                 add_previous_layer_logits: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = LayeredPretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

#        self.bert_model.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout)

        self._add_previous_layer_logits = add_previous_layer_logits
        self._layer_indices = [int(x) for x in layer_indices.split("_")]
        self._sum_weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(i+1)) for i in self._layer_indices])
        self._multitask = multitask
        self._debug = debug

        self._normalize_sum_weights()

        max_layer = max(self._layer_indices)

        # Removing all unused parameters
        self.bert_model.encoder.layer = self.bert_model.encoder.layer[:max_layer+1]

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        self._count_n_layers = CountNLayers(self._layer_indices)
        self._index = index
        self._scaling_temperatures = [float(x) for x in scaling_temperature.split("_")]
        self._temperature_threshold = temperature_threshold

    def _normalize_sum_weights(self):
        for i in range(len(self._sum_weights)):
            self._sum_weights[i] = torch.nn.Parameter(torch.nn.functional.normalize(self._sum_weights[i], p=2, dim=0)) # unit length
        #data[i] = vec / torch.norm(vec)

    def _run_layer(self, input_ids, token_type_ids, input_mask, layer_index, start_index, previous_layer, previous_pooled):
        """Run model on a single layer"""
        encoded_layer, pooled = self.bert_model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=input_mask,
                                                output_all_encoded_layers=True,
                                                layer_index=self._layer_indices[layer_index],
                                                num_predicted_hidden_layers=self._layer_indices[layer_index],
                                                start_index=start_index, previous_layer=previous_layer
                                                )

        # pooled in BERT classification task is the CLS tag (i.e., the first element)
        pooled = self._dropout(pooled)

        if previous_pooled is not None:
            pooled = torch.cat([previous_pooled, pooled])

        return encoded_layer[-1], pooled

