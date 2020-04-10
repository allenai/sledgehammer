"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from myallennlp.dataset_readers.sst_dataset_reader import SSTDatasetReader
from myallennlp.dataset_readers.sst_dataset_reader_oracle import SSTDatasetOracleReader
from myallennlp.dataset_readers.nli_dataset_reader import NLIDatasetReader
from myallennlp.dataset_readers.nli_dataset_reader_oracle import NLIDatasetOracleReader
from myallennlp.dataset_readers.jigsaw_dataset_reader import JigsawDatasetReader
