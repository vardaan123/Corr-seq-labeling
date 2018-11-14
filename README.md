# corr-seq-labelling
Code for Interspeech 2017 paper: Joint Learning of Correlated Sequence Labeling Tasks Using Bidirectional Recurrent Neural Networks [[Link]](https://arxiv.org/pdf/1703.04650.pdf)

**Credits**: [Anirban Laha](https://anirbanl.github.io/), [Vardaan Pahuja](https://vardaan123.github.io/), [Vikas Raykar](https://scholar.google.ca/citations?user=R-Cgh08AAAAJ&hl=en)

## Installation
The TF version used is `0.11.0rc2`. Install the python dependencies using the command `pip install -r requirements.txt`. Install `gunzip` by following the instructions [here](https://www.linuxnix.com/7-linuxunix-gzip-and-gunzip-command-examples/)

## Directory Setup
```
mkdir seq_labelling_logs 
mkdir embed
cd embed
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
```
Note: for all the following groups of 4 commands each, the first is for punctuation best individual model, second is for capitalization best individual model, third is for punctuation best joint model and fourth is for capitalization best joint model.

## Training commands:
```
cd sequence_labelling/
```
### Train on I2 data
```
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/i2data/punc/data.ndp.np.train.txt --trainLabelFilename ../datasets/i2data/punc/data.ndp.train.lab --testFilename ../datasets/i2data/punc/data.ndp.np.test.txt --testLabelFilename ../datasets/i2data/punc/data.ndp.test.lab --devFilename ../datasets/i2data/punc/data.ndp.np.val.txt --devLabelFilename ../datasets/i2data/punc/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model lstm --use_hidden_layer 1 --hidden_dimension 300 --hidden_activation linear --batch_size 64 --num_layers 1
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/i2data/caps/data.capt.ndp.np.train.txt --trainLabelFilename ../datasets/i2data/caps/data.capt.ndp.train.lab --testFilename ../datasets/i2data/caps/data.capt.ndp.np.test.txt --testLabelFilename ../datasets/i2data/caps/data.capt.ndp.test.lab --devFilename ../datasets/i2data/caps/data.capt.ndp.np.val.txt --devLabelFilename ../datasets/i2data/caps/data.capt.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model gru --batch_size 64 --num_layers 2 --rnn_dropout_keep_prob 0.5 --output_dropout_keep_prob 0.5
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/i2data/joint/data.ndp.np.train.txt --trainLabelFilename ../datasets/i2data/joint/data.ndp.train.lab --testFilename ../datasets/i2data/joint/data.ndp.np.test.txt --testLabelFilename ../datasets/i2data/joint/data.ndp.test.lab --devFilename ../datasets/i2data/joint/data.ndp.np.val.txt --devLabelFilename ../datasets/i2data/joint/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 256 --model lstm --use_hidden_layer 1 --hidden_dimension 300 --hidden_activation linear --task_loss_weight 0.9:0.1 --batch_size 64 --num_layers 1
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/i2data/joint/data.ndp.np.train.txt --trainLabelFilename ../datasets/i2data/joint/data.ndp.train.lab --testFilename ../datasets/i2data/joint/data.ndp.np.test.txt --testLabelFilename ../datasets/i2data/joint/data.ndp.test.lab --devFilename ../datasets/i2data/joint/data.ndp.np.val.txt --devLabelFilename ../datasets/i2data/joint/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model lstm --use_hidden_layer 1 --hidden_dimension 300 --hidden_activation linear --task_loss_weight 0.75:0.25 --batch_size 64 --num_layers 2 --rnn_dropout_keep_prob 1.0 --output_dropout_keep_prob 1.0
```

### Train on TED data
```
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/TED_train_corr7/4classes_punct/data.ndp.np.train.txt --trainLabelFilename ../datasets/TED_train_corr7/4classes_punct/data.ndp.train.lab --testFilename ../datasets/TED_2/4classes_punct/data.ndp.np.test.txt --testLabelFilename ../datasets/TED_2/4classes_punct/data.ndp.test.lab --devFilename ../datasets/TED_val_single/4classes_punct/data.ndp.np.val.txt --devLabelFilename ../datasets/TED_val_single/4classes_punct/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 256 --model gru --batch_size 64 --num_layers 1
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/TED_train_corr7/4classes_capt/data.capt.ndp.np.train.txt --trainLabelFilename ../datasets/TED_train_corr7/4classes_capt/data.capt.ndp.train.lab --testFilename ../datasets/TED_2/4classes_capt/data.capt.ndp.np.test.txt --testLabelFilename ../datasets/TED_2/4classes_capt/data.capt.ndp.test.lab --devFilename ../datasets/TED_val_single/4classes_capt/data.capt.ndp.np.val.txt --devLabelFilename ../datasets/TED_val_single/4classes_capt/data.capt.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model gru --batch_size 64 --num_layers 2 --rnn_dropout_keep_prob 0.75 --output_dropout_keep_prob 0.75
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/TED_train_corr7/4classes/data.ndp.np.train.txt --trainLabelFilename ../datasets/TED_train_corr7/4classes/data.ndp.train.lab --testFilename ../datasets/TED_2/4classes/data.ndp.np.test.txt --testLabelFilename ../datasets/TED_2/4classes/data.ndp.test.lab --devFilename ../datasets/TED_val_single/4classes/data.ndp.np.val.txt --devLabelFilename ../datasets/TED_val_single/4classes/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model gru --use_hidden_layer 1 --hidden_dimension 300 --hidden_activation relu --task_loss_weight 0.9:0.1 --batch_size 64 --num_layers 2 --rnn_dropout_keep_prob 0.5 --output_dropout_keep_prob 0.5
python TrainCorrelatedSequenceLabelling.py --save_dir ../seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 --pretrained_embedding_filename ../embed/GoogleNews-vectors-negative300.bin --trainFilename ../datasets/TED_train_corr7/4classes/data.ndp.np.train.txt --trainLabelFilename ../datasets/TED_train_corr7/4classes/data.ndp.train.lab --testFilename ../datasets/TED_2/4classes/data.ndp.np.test.txt --testLabelFilename ../datasets/TED_2/4classes/data.ndp.test.lab --devFilename ../datasets/TED_val_single/4classes/data.ndp.np.val.txt --devLabelFilename ../datasets/TED_val_single/4classes/data.ndp.val.lab --early_stopping 1 --max_epochs 100 --rnn_size 512 --model lstm --use_hidden_layer 1 --hidden_dimension 300 --hidden_activation relu --task_loss_weight 0.75:0.25 --batch_size 64 --num_layers 1 --rnn_dropout_keep_prob 0.75 --output_dropout_keep_prob 0.75
```

## Compute label predictions
```
cd sequence_labelling/
```
### I2 data Reference
```
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0
```

### Test-dataset-2 Reference
```
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5
python evaluateCorrelatedSequenceLabels.py ../seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75
```

### Test-dataset-1 Reference
```
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 ../datasets/TED_3/4classes/data.ndp.np.test.txt ../datasets/TED_3/4classes/data.ndp.test.lab ted3
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 ../datasets/TED_3/4classes_capt/data.capt.ndp.np.test.txt ../datasets/TED_3/4classes_capt/data.capt.ndp.test.lab ted3
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 ../datasets/TED_3/4classes/data.ndp.np.test.txt ../datasets/TED_3/4classes/data.ndp.test.lab ted3
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 ../datasets/TED_3/4classes/data.ndp.np.test.txt ../datasets/TED_3/4classes/data.ndp.test.lab ted3
```

### Test-dataset-2 ASR
```
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 ../datasets/TED_2_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_2_ASR/4classes/data.ndp.test.lab ted2_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 ../datasets/TED_2_ASR/4classes_capt/data.capt.ndp.np.test.txt ../datasets/TED_2_ASR/4classes_capt/data.capt.ndp.test.lab ted2_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 ../datasets/TED_2_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_2_ASR/4classes/data.ndp.test.lab ted2_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 ../datasets/TED_2_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_2_ASR/4classes/data.ndp.test.lab ted2_asr
```

### Test-dataset-1 ASR
```
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 ../datasets/TED_3_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_3_ASR/4classes/data.ndp.test.lab ted3_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 ../datasets/TED_3_ASR/4classes_capt/data.capt.ndp.np.test.txt ../datasets/TED_3_ASR/4classes_capt/data.capt.ndp.test.lab ted3_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 ../datasets/TED_3_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_3_ASR/4classes/data.ndp.test.lab ted3_asr
python evaluateCorrelatedSequenceLabelsTest.py ../seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 ../datasets/TED_3_ASR/4classes/data.ndp.np.test.txt ../datasets/TED_3_ASR/4classes/data.ndp.test.lab ted3_asr
```

## Computation of evaluation metrics

### I2 data Reference
```
python sequence_labelling/metrics_full_correlated.py datasets/i2data/punc/data.ndp.test.lab seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1/test_predictions_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/i2data/caps/data.capt.ndp.test.lab seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5/test_predictions_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/i2data/joint/data.ndp.test.lab seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1/test_predictions_task0.tsv seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1/test_predictions_task1.tsv
python sequence_labelling/metrics_full_correlated.py datasets/i2data/joint/data.ndp.test.lab seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0/test_predictions_task0.tsv seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0/test_predictions_task1.tsv
```

### Test-dataset-2 Reference
```
python sequence_labelling/metrics_full_correlated.py datasets/TED_2/4classes_punct/data.ndp.test.lab seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1/test_predictions_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_2/4classes_capt/data.capt.ndp.test.lab seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75/test_predictions_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_2/4classes/data.ndp.test.lab seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/test_predictions_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/test_predictions_task1.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_2/4classes/data.ndp.test.lab seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/test_predictions_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/test_predictions_task1.tsv
```

### Test-dataset-1 Reference
```
python sequence_labelling/metrics_full_correlated.py datasets/TED_3/4classes_punct/data.ndp.test.lab seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1/ted3_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_3/4classes_capt/data.capt.ndp.test.lab seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75/ted3_task0.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_3/4classes/data.ndp.test.lab seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted3_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted3_task1.tsv
python sequence_labelling/metrics_full_correlated.py datasets/TED_3/4classes/data.ndp.test.lab seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted3_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted3_task1.tsv
```

### Test-dataset-2 ASR
```
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_2/4classes_punct/data.ndp.test.lab datasets/TED_2_ASR/ted2_asr_punct_slots.txt seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1/ted2_asr_task0.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_2/4classes_capt/data.capt.ndp.test.lab datasets/TED_2_ASR/ted2_asr_punct_slots.txt seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75/ted2_asr_task0.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_2/4classes/data.ndp.test.lab datasets/TED_2_ASR/ted2_asr_punct_slots.txt seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted2_asr_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted2_asr_task1.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_2/4classes/data.ndp.test.lab datasets/TED_2_ASR/ted2_asr_punct_slots.txt seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted2_asr_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted2_asr_task1.tsv
```

### Test-dataset-1 ASR
```
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_3/4classes_punct/data.ndp.test.lab datasets/TED_3_ASR/ted3_asr_punct_slots.txt seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1/ted3_asr_task0.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_3/4classes_capt/data.capt.ndp.test.lab datasets/TED_3_ASR/ted3_asr_punct_slots.txt seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75/ted3_asr_task0.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_3/4classes/data.ndp.test.lab datasets/TED_3_ASR/ted3_asr_punct_slots.txt seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted3_asr_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5/ted3_asr_task1.tsv
python sequence_labelling/metrics_full_correlated_asr.py datasets/TED_3/4classes/data.ndp.test.lab datasets/TED_3_ASR/ted3_asr_punct_slots.txt seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted3_asr_task0.tsv seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75/ted3_asr_task1.tsv
```
