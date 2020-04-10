"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from myallennlp.dataset_readers.classification_dataset_reader import ClassificationDatasetReader
from myallennlp.dataset_readers.classification_dataset_reader_oracle import ClassificationDatasetOracleReader
from myallennlp.dataset_readers.nli_dataset_reader import NLIDatasetReader
from myallennlp.dataset_readers.nli_dataset_reader_oracle import NLIDatasetOracleReader
