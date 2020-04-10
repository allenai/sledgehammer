from allennlp.data.fields import ArrayField, TextField, LabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from typing import *
from allennlp.data.token_indexers import TokenIndexer
from overrides import overrides
from allennlp.data.tokenizers import Token
import numpy as np
from allennlp.data import Instance
import pandas as pd
import torch

label_cols = ['accuracy']

@DatasetReader.register("sst_dataset_reader_oracle")
class SSTDatasetOracleReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                         token_indexers: Dict[str, TokenIndexer] = None,
                         max_seq_len: Optional[int]=100, testing=False) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len
        self.testing=testing,
        self.label_cols = label_cols

    @overrides
    def text_to_instance(self, tokens: List[Token],
                         instance_id: int=-1,
                         layer_index: int=-1,
                         labels: int=0) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)

        label_field = LabelField(labels)
#        print(instance_id, labels, tokens)
        gold_layer_field = MetadataField(layer_index)
        #print(id, ta, label_field)
        id_field = MetadataField(instance_id)
                
        fields = {"tokens": sentence_field, 'label': label_field, "instance_id": id_field,
                                                    "gold_layer": gold_layer_field}

        return Instance(fields)
                                                                                                            
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='ISO-8859-1') as ifh:
#        with open(file_path) as ifh:
#        if self.testing: df = df.head(1000)
            for (i, l) in enumerate(ifh):
                fields = l.rstrip().split("\t")
                if len(fields) < 3:
                    continue
                yield self.text_to_instance(
                    [Token(x) for x in self.tokenizer(fields[2])],
                    i,
                    int(fields[1]),
                    fields[0]
                )

