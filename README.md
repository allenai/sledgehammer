# Sledgehammer
An improved  method for contextual representation fine-tuning which, during  inference, allows for an early (and fast) "exit" from neural network calculations for  simple  instances  and  late  (and  accurate) exit  for  hard  instances. Based on ["The Right Tool for the Job: Matching Model and Instance Complexities"](https://arxiv.org/abs/???) by Roy Schwartz, Gabriel Stanovsky, Swabha Swayamdipta, Jesse Dodge and Noah A. Smith, ACL 2020


## Setup

The code is implemented in python3.6 using AllenNLP. To run it, please install the requirements.txt file:

```pip install -r requirements.txt```

All commands below are found in the `scripts` directory. Run each with `-h` to see additional options.

## Fine-Tuning a model for a downstream task
To fine-tune a pretrained model, run the `train_model.py` command. The script samples a learning rate between `[2e-5, 3e-5, 5e-5]` and a random seed in the range (0,100000). It can be used to run multiple random search experiments over these parameters (see `-n` flag below).

```python scripts/train_model.py
-t bert_type: One of `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`, `bert-large-cased`, `bert-base-multilingual-uncased`, `bert-base-multilingual-cased`, `bert-base-chinese`.
-l layer_indices: '_'-separated list indices of layers to add classifiers to (each between [0-(num_layers-1)])
-w work_dir: working directory to store output experiments in
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
To calibrate a trained model, run the `run_calibration.py` command. The script gets as input a saved model and the development set, and prints the temperatures of all classifiers ('_' separated):

```
python scripts/run_calibration.py -m <saved model file (.th file)> -v <development set file> -u <CUDA device (or -1 for CPU)>
```

## Evaluation

To evaluate our model, run the `run_evaluation.py` script. The script gets as input the output of the calibration model (a '_'-separated list of temperatures, one per classification layer):

```
python scripts/run_evaluation.py -t <calibration_temperatures (one per classification layer, '_' separated)> -u <CUDA device (or -1 for CPU)> -v <dev set file> -o <output file> -m <saved model file (.th file)>
```



To run the full pipeline, run the following:

1. Run grid search to train $k$ models on each layer considered. E.g.,

`python scripts/multiloss_grid_search.py 
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size (default: 72)
  -c NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to run (default: 2)
  -x MAX_PIECES, --max_pieces MAX_PIECES
                        Maximum number of word pieces for BERT (default: 512)
  -t BERT_TYPE, --bert_type BERT_TYPE
                        Bert type (bert-{base,large}-{cased,uncased})
                        (default: bert-large-uncased)

  -n N_TESTS, --n_tests N_TESTS
                        Number of grid search experiments to run (default: 1)
  -l LAYER_INDICES, --layer_indices LAYER_INDICES
                        Indices of layers to train classifiers for (default: 23)

  -d DATASET, --dataset DATASET
                        Dataset to work with (default: None)
  -i, --nli             Is this an NLI experiment? (if not, it's text_cat) (default: False)

  -w WORK_DIR, --work_dir WORK_DIR
                        Working directory. Should contain a directory for the
                        bert_vocab, which contains another directory for the
                        dataset (default:
                        /net/nfs.corp/allennlp/roys/work/sledgehammer/models/)
  --data_dir DATA_DIR   Dataset directory. Should contain 'text_cat' and/or
                        'nli' folders, containing a directory for the dataset,
                        which contains three files: train, dev and test
                        (default: /Users/roysch/resources/)
E.g., `python scripts/multiloss_grid_search.py -t bert-base-uncased -l 0,4,12,23 -d amazon_reviews -w <work dir> --data_dir <data dir>`

2. Select the best model for each layer, and extract the linear layer:
age: scripts/extract_linear_layer.py <if> <of>
`scripts/extract_linear_layer.py <input file> <output file>`
E.g., `python scripts/extract_linear_layer.py ~/work/sledgehammer/models/amazon_reviews/learned_23_predicted_23_13/{model.tar.gz,linear_layer.th}`

3. Generate predictions for the selected model:
`scripts/gen_predictions.sh <work dir>`
E.g., `scripts/gen_predictions.sh ~/work/sledgehammer/models/amazon_reviews/learned_23_predicted_23_13/`

To run the simulation and get plots:

    ./scripts/run_simulation.sh
