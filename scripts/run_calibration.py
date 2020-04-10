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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main():
    parser = arg_parser()

    args = parser.parse_args()
    
    temp_value = "1._1._1._1."
    optimizer = args.optimizer
    n_epochs = 10000

    if optimizer == 'adam':
        optimizer = Adam_optimizer(n_epochs)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS_optimizer(n_epochs)
    else:
        print("Illegal optimizer {}, must be one of (adam, lbfgs)")
        return -2

    temp_value_list = [float(x) for x in temp_value.split("_")]
    n_layers = len(temp_value_list)

    model_file = args.model_file
    serialization_dir = "/".join(args.model_file.split("/")[:-1])
    evaluation_data_path = args.dev_file

    cuda_device=args.cuda_device
    
    # output file
    overrides = "{ iterator: {batch_size: 1}, model: {temperature_threshold: 1, scaling_temperature: '"+temp_value+"'}}"
    print(overrides)
    archive = load_archive(serialization_dir, cuda_device, overrides, model_file)
    config = archive.config

    model = archive.model

    if cuda_device >= 0:
        model = model.cuda()
    model._scaling_temperatures = temp_value_list
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

    print("\n\n######################################\n")
    print("#  Finished computing temperatures.  #\n")
    print("######################################\n")
    print("Values are: {}".format("_".join([str(x) for x in temp])))

    return 0


def arg_parser():
    """Extracting CLI arguments"""
    p = ArgumentParser(add_help=False)

    p.add_argument("-u", "--cuda_device", help="CUDA device (or -1 for CPU)", type=int, default=0)
    p.add_argument('-m', '--model_file', help="Model file",  type=str, required=True)
    p.add_argument('-v', '--dev_file', help="Development set file",  type=str, required=True)
    p.add_argument('-o', '--optimizer', help="Optimizer (adam or lbfgs)",  type=str, default="adam")


    return  ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[p])


if __name__ == '__main__':
    sys.exit(main())



