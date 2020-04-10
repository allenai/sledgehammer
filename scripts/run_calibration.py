#!/usr/bin/env python

import os
import sys
from temperature_scaling.temperature_scaling import ModelWithTemperature,set_temperature, Adam_optimizer, LBFGS_optimizer, Optimizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.iterators import DataIterator
from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from myallennlp.data.token_indexers import *
from myallennlp.modules.token_embedders import *
from myallennlp.pytorch_pretrained_bert import *
from myallennlp.models.multiloss_bert_for_classification import MultilossBertForClassification
from myallennlp.dataset_readers.sst_dataset_reader import SSTDatasetReader

def main(args):
    temp_value = "1.,1.,1.,1."
    optimizer = "adam"
    if len(args) < 3:
        print("Usage: {} <model file> <dataset file> <optimizer = adam (or lbfgs)>".format(args[0], temp_value))
        return -1
    elif len(args) > 3:
        optimizer = args[4]

    n_epochs = 10000

    if optimizer == 'adam':
        optimizer = Adam_optimizer(n_epochs)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS_optimizer(n_epochs)
    else:
        print("Illegal optimizer {}, must be one of (adam, lbfgs)")
        return -2

    temp_value = [float(x) for x in temp_value.split(",")]
    n_layers = len(temp_value)

    d = args[1]

    if d[-1] != '/':
        d += '/'

    evaluation_data_path = args[2]
    epoch = args[3]

    cuda_device=0
    
    # output file
    if epoch == '-1':
        model = evaluation_data_path
        # Load vocabulary from file
        # If the config specifies a vocabulary subclass, we need to use it.
        generator_tqdm = None
    else:
        overrides = '{ iterator: {batch_size: 1}, model: {temperature_threshold: 1}}'
        archive = load_archive(d, cuda_device, overrides, d+'model_state_epoch_'+epoch+'.th')
        config = archive.config

        model = archive.model
        model = model.cuda()
        model._scaling_temperatures = temp_value
        model.eval()
        vocab = model.vocab

        validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
        if validation_dataset_reader_params is not None:
            dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
        else:
            dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
    
        instances = dataset_reader.read(evaluation_data_path)

        iterator_params = config.pop("iterator")

        data_iterator = DataIterator.from_params(iterator_params)

        data_iterator.index_with(vocab)

        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
        
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        temp = set_temperature(model, n_layers, optimizer, generator_tqdm, cuda_device)

    print(temp)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
