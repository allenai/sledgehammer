{
  "random_seed": std.parseInt(std.extVar("SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "dataset_reader": {
    "type": "classification_dataset_reader",
     "token_indexers": {
      "bert": {
          "type": "nodebug-bert-pretrained",
          "pretrained_model": std.extVar("BERT_TYPE"),
          "do_lowercase": std.extVar("IS_LOWERCASE"),
          "use_starting_offsets": true,
	  "max_pieces": std.extVar("MAX_PIECES"), 
//	  "truncate_long_sequences": false
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("DEV_PATH"),
  "test_data_path": std.extVar("TEST_PATH"),
  "evaluate_on_test": true,
  "model": {
    "type": "multiloss_bert_for_classification",
    "bert_model": std.extVar("BERT_TYPE"),
    "trainable": true,
    "dropout": std.extVar("DROPOUT"),
    "layer_indices": std.extVar("LAYER_INDICES"),
    "scaling_temperature": std.extVar("SCALING_TEMPERATURE"),
    "temperature_threshold": std.extVar("TEMPERATURE_THRESHOLD"),
    "add_previous_layer_logits": std.extVar("ADD_PREVIOUS_LAYER_LOGITS"),
    "multitask": std.extVar("MULTITASK")
  },
  "validation_iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 1
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": std.extVar("BATCH_SIZE")
  },
  "trainer": {
    "num_epochs": std.extVar("NUM_EPOCHS"),
    "patience": 5,
    "validation_metric": "+accuracy",
    "cuda_device": std.extVar("CUDA_DEVICE"),
    "optimizer": {
        "type": "adam",
        "lr": std.extVar("LEARNING_RATE")
    },
  }
}

