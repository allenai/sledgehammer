from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from typing import *
from allennlp.data.token_indexers import TokenIndexer
from overrides import overrides
from allennlp.data.tokenizers import Token
import numpy as np
from allennlp.data import Instance
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]

class JigsawDatasetReader(DatasetReader):
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
    def text_to_instance(self, tokens: List[Token], id: str=None,
                         labels: np.ndarray=None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        id_field = MetadataField(id)
        fields["id"] = id_field
                                                            
        if labels is None:
            labels = np.zeros(len(label_cols))
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
                                                                                                            
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        if self.testing: df = df.head(1000)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["comment_text"])],
                row["id"], row[label_cols].values,
            )

