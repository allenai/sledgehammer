#!/usr/bin/env python

import sys
import os
import random
import copy
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# PYTHON_DIR="/".join(os.environ['CONDA_EXE'].split("/")[:-2])+'/envs/allennlp_0.8.4/bin/'
exit_threshold=0.9

def main():
        parser = arg_parser()

        args = parser.parse_args()

        lrs = [2e-5, 3e-5, 5e-5]
        dropout = [0.1, 0.1]
        
        layer_indices = args.layer_indices
        dont_delete = len(layer_indices.split("_")) > 1
        n_test = args.n_tests
        dataset = args.dataset
        start = args.start
        is_lowercase = args.bert_type[-7:] == 'uncased'

        cwd = os.getcwd()+"/"

        training_config_file = cwd+"training_config/sledgehammer_bert_classification.jsonnet"
        base_path = args.data_dir+"/text_cat/" 
        if args.nli:
            training_config_file = cwd+"training_config/sledgehammer_bert_nli.jsonnet"
            base_path = args.data_dir+"/nli/" 
             

        slurm = args.slurm

        extra_args = ""

        if slurm is None:
            os.environ["BERT_TYPE"] = args.bert_type
            os.environ["IS_LOWERCASE"] = str(is_lowercase).lower()
            os.environ["TRAIN_PATH"] = base_path+dataset+"/train"
            os.environ["DEV_PATH"] = base_path+dataset+"/dev"
            os.environ["TEST_PATH"] = base_path+dataset+"/test"
            os.environ["LAYER_INDICES"] = layer_indices
            # @todo change me back to 0
            os.environ["CUDA_DEVICE"] = str(args.cuda_device)
            os.environ["SCALING_TEMPERATURE"] = "_".join(["1" for i in range(len(layer_indices.split("_")))])
            os.environ["BATCH_SIZE"] = str(args.batch_size)
            os.environ["MAX_PIECES"] = str(args.max_pieces)
            os.environ["TEMPERATURE_THRESHOLD"] = str(exit_threshold)
            os.environ["ADD_PREVIOUS_LAYER_LOGITS"] = 'false'
            os.environ["MULTITASK"] = 'false'
            os.environ["NUM_EPOCHS"] = str(args.num_epochs)
        else:
            extra_args = "--export BERT_TYPE={},IS_LOWERCASE={},TRAIN_PATH={},DEV_PATH={},TEST_PATH={},LAYER_INDICES={},CUDA_DEVICE={},SCALING_TEMPERATURE={},BATCH_SIZE={},MAX_PIECES={},TEMPERATURE_THRESHOLD={},ADD_PREVIOUS_LAYER_LOGITS={},MULTITASK={},NUM_EPOCHS={}".format(args.bert_type,str(is_lowercase).lower(),base_path+dataset+"/train",base_path+dataset+"/dev",base_path+dataset+"/test","'"+layer_indices+"'",0,"'"+"_".join(["1"  for i in range(len(layer_indices.split("_")))])+"'",args.batch_size,args.max_pieces,exit_threshold,'false','false',args.num_epochs)  

        for i in range(start, n_test):
            #lr = str(10**random.uniform(lrs[0], lrs[1]))
            lr = str(lrs[random.randint(0, len(lrs))-1])
            dr = str(random.uniform(dropout[0], dropout[1]))
            seed = str(random.randint(0,100000))
            local_dir = args.work_dir+args.bert_type+"/"+dataset+"/experiment_{}_{}/".format(layer_indices, i)
            local_extra_args = copy.copy(extra_args)
            allennlp_cmd = "allennlp train {} --serialization-dir {} --include-package allennlp_overrides -f".format(training_config_file, local_dir)
            if slurm is None:
                os.environ["SEED"] = seed
                os.environ["PYTORCH_SEED"] = seed
                os.environ["NUMPY_SEED"] = seed
                os.environ["DROPOUT"] = dr
                os.environ["LEARNING_RATE"] = lr
                cmd = allennlp_cmd
            else:
                local_extra_args += ",SEED={},PYTORCH_SEED={},NUMPY_SEED={},DROPOUT={},LEARNING_RATE={}".format(seed,seed,seed,dr,lr)
                cmd = "srun -p allennlp_hipri -w {} --gpus=1 {} {}".format(slurm, local_extra_args, allennlp_cmd)

            print(cmd)

            return_value = subprocess.call(cmd, shell=True)

            if return_value != 0:
                for j in range(200):
                    if not dont_delete:
                        f = "{}/model_state_epoch_{}.th".format(local_dir, j)
                        rm_if_exists(f)
                    f = "{}/training_state_epoch_{}.th".format(local_dir, j)
                    rm_if_exists(f)

                f = local_dir+"/best.th"
                rm_if_exists(f)

                # If we are not deleting intermediate models, we don't need the final model.tar.gz file
                if dont_delete:
                    f = local_dir+"/model.tar.gz"
                    rm_if_exists(f)
        

        return 0

def rm_if_exists(f):
    if os.path.exists(f):
        os.remove(f)
        return 1
    else:
        return 0



def arg_parser():
    """Extracting CLI arguments"""
    p = ArgumentParser(add_help=False)

    p.add_argument("-b", "--batch_size", help="Batch size", type=int, default=72)
    p.add_argument("-s", "--start",
                   help="First experiment index to run",
                   type=int, default=0)
    p.add_argument("-t", "--bert_type", help="Bert type (bert-{base,large}-{cased,uncased})", type=str,
                   default='bert-large-uncased')
    p.add_argument("-n", "--n_tests", help="Number of grid search experiments to run", type=int, default=1)
    p.add_argument("-x", "--max_pieces", help="Maximum number of word pieces for BERT", type=int, default=512)
    p.add_argument("-c", "--num_epochs", help="Number of epochs to run", type=int, default=2)
    p.add_argument("-l", "--layer_indices", help="Indices of layers to train classifiers for", type=str, default="23")
    p.add_argument("-d", "--dataset", help="Dataset to work with", required=True)
    p.add_argument("-i", "--nli", help="Is this an NLI experiment? (if not, it's text_cat)", action='store_true')
    p.add_argument("-r", "--slurm", help="Run jobs on SLURM using this server", type=str)
    p.add_argument('-w', '--work_dir', help="Working directory. Should contain a directory for the bert_type, which contains another directory for the dataset", type=str, default="")
    p.add_argument('--data_dir', help="Dataset directory. Should contain 'text_cat' and/or 'nli' folders, containing a directory for the dataset, which contains three files: train, dev and test",  type=str, required=True)
    p.add_argument("-u", "--cuda_device", help="CUDA device (or -1 for CPU)", type=int, default=0)


    return  ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[p])


if __name__ == '__main__':
    sys.exit(main())

