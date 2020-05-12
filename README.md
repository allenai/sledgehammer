# Sledgehammer
An improved  method for contextual representation fine-tuning which, during  inference, allows for an early (and fast) "exit" from neural network calculations for  simple  instances  and  late  (and  accurate) exit  for  hard  instances. Based on ["The Right Tool for the Job: Matching Model and Instance Complexities"](http://arxiv.org/abs/2004.07453) by Roy Schwartz, Gabriel Stanovsky, Swabha Swayamdipta, Jesse Dodge and Noah A. Smith, ACL 2020.


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
python scripts/train_model.py \
-t bert-base-uncased \
-l 0_3_5_11 \
--data_dir data_dir \
-d imdb \
-w <working directory>
```

### Datasets
This package currently only supports text classification and NLI datasets.
To download the datasets experimented with in our paper (AG news, IMDB, SST, SNLI and MultiNLI), 
run the following script:

```
scripts/download_data.sh <output directory>
```

E.g., 
```
scripts/download_data.sh ./resources/
```

This script will download the 5 datasets and put them in `<output directort>`.
This directory will contain two sub-directories: `text_cat` (for text classification datasets) and `nli` (for NLI datasets).
Inside each directory, a directory will be created for each dataset (e.g., `text_cat/imdb`, `nli/snli`).
Inside the dataset directories, there will be three files: `train`, `dev`, and `test`.
Each of these files would contain one line per instance in the following format:

```
label	  text
```

E.g.,
```
1   The movie was great
0   I didn't like the book
```

Note that some of the datasets are large, so downloading them may take a few minutes.
To experiment with additional datasets, download them and put them in the corresponding directories as explained above.

## Temperature Calibration
To calibrate a trained model, run the `run_calibration.py` command. The script gets as input a saved model and the development set, and prints the temperatures of all classifiers ('_' separated). It then runs adam (LBFGS optimizer also available with the `-o` flag). 
The code builds on a modified version of https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

```
python scripts/run_calibration.py \
-m <saved model file (.th file)> \
-v <development set file>
```


For instance, here as an example of the last few lines of a successful output of the script:


```
...
######################################

#  Finished computing temperatures.  #

######################################

Values are: 0.5930193662643433_1.159342885017395_1.1623098850250244_1.2985113859176636
```

The resulting values (`0.5930193662643433_1.159342885017395_1.1623098850250244_1.2985113859176636` in this example) should be copied and fed to the evaluation script (see below).

## Evaluation

To evaluate our model, run the `run_evaluation.py` script. The script gets as input the output of the calibration model (the last line of the output of the `run_calibration.py` script, see above): a `'_'`-separated list of temperatures, one per classification layer. 
It also gets a confidence threshold (in the range [0-1]) which controls the speed/accuracy tradeoff. Lower values favor speed, while higher values favor accuracy. The model's output is saved in a file called <output_file>_<confidence_threshold>. The model's speed (seconds) is shown in the script's output.

```
python scripts/run_evaluation.py  \
-t <output from the calibration script: one per classification layer, '_' separated> \
-c <confidence_threshold value> \
-v <dev set file> \
-o <output file> \
-m <saved model file (.th file)>
```

## Trouble shooting
If training crashes due to CUDA memory errors, try reducing the batch size (`-b` flag).

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

