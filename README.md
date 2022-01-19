# Automatic Speech Recognition for Luganda


We trained the [Automatic Speech Recognition for Luganda](https://huggingface.co/indonesian-nlp/wav2vec2-luganda) model 
as part of [Mozilla Luganda Automatic Speech Recognition](https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition/).
Our model achieve the Word Error Rate (WER) of **7.37%** on Mozilla Common Voice version 6.1
and **7.53%** on Mozilla Common Voice version 7.0.

We use several python scripts to do the training/fine-tuning, evaluation and the creation of submission file:
- run_finetuning.py
- run_evaluation.py
- run_submission.py 

We prepared a [jupyter notebook](Luganda_ASR.ipynb) to run all these tasks in Google Colab.

## Dataset
We trained the model using [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
dataset version 7.0 for Luganda.
The test dataset can be downloaded from [Mozilla Luganda - ASR dataset](https://zindi.africa/competitions/mozilla-luganda-automatic-speech-recognition/data).

## Model Training

We have fine-tuned the Facebook Wav2Vec2 model with the Luganda Common Voice dataset and stored it in https://huggingface.co/indonesian-nlp/wav2vec2-luganda. The model has been trained for 200 epochs in 4 hours and 30 minutes using 8 GPUs.

We use the script run_finetuning.py to train the model.
Due to resource and time limitation in Google Colab,
we skip the model training here, but we run the evaluation and creation of the submission file. However, feel free to run following command for testing purpose:

```!python run_finetuning.py finetuning_common_voice_1epoch.json```

It will run the training for only one epoch which will take around 2 hours in Google Colab.

### Usage
For training using single GPU
``` 
% python run_finetuning.py <argument json file>
``` 
For training using multi GPUs, for example 8 GPUs.
``` 
% python -m torch.distributed.launch --nproc_per_node=8 run_finetuning.py <argument json file>
``` 
Our model "indonesian-nlp/wav2vec2-luganda" has been trained using following command:
``` 
% python -m torch.distributed.launch --nproc_per_node=8 run_finetuning.py finetuning_common_voice.json
```

List of possible arguments:
``` 
% python run_finetuning.py -h
``` 

## Model Evaluation
To test our fine-tuned model, we evaluate it with test split of Mozilla Common Voice dataset version 6.1 and 7.0. The evaluation also uses the Ken Language Model (KenLM) 5gram.bin we created from the text of Common Voice 7.0.


### Usage
Following is the command to evaluate test split of Mozilla Common Voice dataset using our model "indonesian-nlp/wav2vec2-luganda" and using the KenLM:
``` 
% python run_evaluation.py -m indonesian-nlp/wav2vec2-luganda -n common_voice -c lg -k 5gram.bin --test_pct 100
```
List of possible arguments:
``` 
% python run_evaluation.py -h
usage: run_evaluation.py [-h] -m MODEL_NAME -n NAME -c CONFIG_NAME [-d DATA_DIR] [-b BATCH_SIZE] [-k KENLM] [--num_workers NUM_WORKERS] [-w BEAM_WIDTH] [--test_pct TEST_PCT] [--cpu]

optional arguments:
-h, --help            show this help message and exit
-m MODEL_NAME, --model_name MODEL_NAME
The wav2vec2 model name
-n NAME, --name NAME  The name of dataset
-c CONFIG_NAME, --config_name CONFIG_NAME
The config name of the dataset
-d DATA_DIR, --data_dir DATA_DIR
The directory contains the dataset
-b BATCH_SIZE, --batch_size BATCH_SIZE
Batch size
-k KENLM, --kenlm KENLM
Path to KenLM model
--num_workers NUM_WORKERS
KenLM's number of workers
-w BEAM_WIDTH, --beam_width BEAM_WIDTH
KenLM's beam width
--test_pct TEST_PCT   Percentage of the test set
--cpu                 Force to use CPU

```

## Submission File Creation

We will create the submission file "submissions/luganda-asr.csv"
using the ASR model "indonesian-nlp/wav2vec2-luganda" on the test set
provided by Zindi.
It takes around 40 minutes

### Usage
```
$ python run_submission.py -h
usage: run_submission.py [-h] -m MODEL_NAME -d DATA_DIR -o OUTPUT_FILE [-b BATCH_SIZE] [-k KENLM] [-n NUM_WORKERS] [-w BEAM_WIDTH] [--test_pct TEST_PCT]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        The wav2vec2 model name
  -d DATA_DIR, --data_dir DATA_DIR
                        The directory contains the Zindi dataset (Train.csv, Test.csv and validated_dataset)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        The file name of the prediction result
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -k KENLM, --kenlm KENLM
                        Path to KenLM model
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        KenLM's number of workers
  -w BEAM_WIDTH, --beam_width BEAM_WIDTH
                        KenLM's beam width
  --test_pct TEST_PCT   Percentage of the test set
```