"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from myallennlp.models.multi_layered_bert_for_classification import MultiLayeredBertForClassification
from myallennlp.models.layered_bert_for_classification import LayeredBertForClassification
from myallennlp.models.multiloss_bert_for_classification import MultilossBertForClassification
from myallennlp.models.multiloss_bert import MultilossBert
