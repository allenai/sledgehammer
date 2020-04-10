"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from myallennlp.pytorch_pretrained_bert.modeling import LayeredBertModel
from myallennlp.pytorch_pretrained_bert.tokenization import BertTokenizer
