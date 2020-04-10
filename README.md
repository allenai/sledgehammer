# Sledgehammer
An improved  method for contextual representation fine-tuning which, during  inference, allows for an early (and fast) "exit" from neural network calculations for  simple  instances  and  late  (and  accurate) exit  for  hard  instances. Based on ["The Right Tool for the Job: Matching Model and Instance Complexities"](https://arxiv.org/abs/???) by Roy Schwartz, Gabriel Stanovsky, Swabha Swayamdipta, Jesse Dodge and Noah A. Smith, ACL 2020.


## Setup

The code is implemented in python3.6 using AllenNLP. To run it, please install the requirements.txt file:

```pip install -r requirements.txt```

All commands below are found in the `scripts` directory. Each is run on GPU (with device 0 by default). A different GPU can be configured in all scripts with the `-u` flag, with `-1` running on CPU. Run each script with `-h` to see additional options.

## Fine-Tuning a model for a downstream task
To fine-tune a pretrained model, run the `train_model.py` command. The script samples a learning rate between `[2e-5, 3e-5, 5e-5]` and a random seed in the range (0,100000). It can be used to run multiple random search experiments over these parameters (see `-n` flag below).

```
python scripts/train_model.py
-t bert_type: One of bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.
-l layer_indices: '_'-separated list indices of layers to add classifiers to (each between [0-(num_layers-1)])
-w work_dir: working directory to store output experiments in. Inside it, the following folders will be created:  <bert_type>/<dataset>/experiment_<layer_indices>_<experiment_index>/
-i: an NLI experiment (if false, a text classification experiment)
-d dataset_name_: dataset name to work with
-n <n_runs = 1>: number of experiments to run
--data_dir data_dir: directory where data is stored. Structure should be:
data_dir/text_cat/<dataset name>/{train,dev,test}
data_dir/nli/<dataset name>/{train,dev,test}
```

E.g., 

```
python scripts/train_model.py
-t bert-base-uncased
-l 0_3_5_11
-w ~/resources/
-d imdb
--data_dir <working directory>
```

## Temperature Calibration
To calibrate a trained model, run the `run_calibration.py` command. The script gets as input a saved model and the development set, and prints the temperatures of all classifiers ('_' separated). It then runs adam (LBFGS optimizer also available with the `-o` flag). 
The code builds on a modified version of https://github.com/HMJiangGatech/gat_graphsage_conf/blob/master/temperature_scaling.py

```
python scripts/run_calibration.py
-m <saved model file (.th file)>
-v <development set file>
```

## Evaluation

To evaluate our model, run the `run_evaluation.py` script. The script gets as input the output of the calibration model (the last line of the output of the `run_calibration.py` script): a `'_'`-separated list of temperatures, one per classification layer. 
It also gets a confidence threshold (in the range [0-1]) which controls the speed/accuracy tradeoff. Lower values favor speed, while higher values favor accuracy. The model's output is saved in a file called <output_file>_<confidence_threshold>. The model's speed (seconds) is shown on the scripts' output.

```
python scripts/run_evaluation.py 
-t <calibration_temperatures (one per classification layer, '_' separated)>
-c <confidence_threshold value>
-v <dev set file> 
-o <output file>
-m <saved model file (.th file)>
```

## References
If you make use if this code, please cite the following paper:

```latex

@inproceedings{Schwartz:2020,
  author={Schwartz, Roy and Stanovsky, Gabriel and Swayamdipta, Swabha and Jesse Dodge and Smith, Noah A.},
  title={The Right Tool for the Job: Matching Model and Instance Complexities},
  booktitle={Proc. of ACL},
  year={2020}
}
```

## Contact

For inquiries, please file an [issue](https://github.com/allenai/sledgehammer/issues) or email roys@allenai.org.
