# Instructions for using sequence classification using RNN/CNN

This will have representative scripts for training and prediction.

## Single sequence classification

Single sequence that is context-free classification

### RNN:
```
python TrainRNNSequenceClassification.py --rnn_size 100 --num_layers 2 --filename <> --model gru --max_epochs 50 --max_seq_length 50 --save_dir <> --pretrained_embedding_filename /dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin
```

### CNN:
```
python TrainCNNSequenceClassification.py --num_filters 20 --filter_sizes 3,4,5 --l2_reg_lambda 0.01 --train_file <> --max_epochs 50 --max_seq_length 50 --save_dir <> --pretrained_embedding_filename /dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin 
```

## BiSequence classification

Context-dependent classification where the two sequences are context and sentence

### Training script sample (change script appropriately):
Architecture Concat-CNN-CNN:
```
python TrainCNN.py --task debater --filename ../datasets/task_b-88_claim_sentence_examples.csv --pretrained_embedding_filename /dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin --max_context_seq_length 14 --max_sentence_seq_length 60 --max_epochs 30 --architecture cnnconcat --save_dir claim_concat_cnn_cnn_full_lomo_1 --l2_reg_lambda 0.01 --context_filter_sizes 3,4 --sentence_filter_sizes 3,4 --context_num_filters 64 --sentence_num_filters 64 --motion_id 1
```

Architecture ConditionalStateInput-RNN-RNN:
```
python TrainRNNBiSequenceClassification.py --task debater --filename ../datasets/task_c-68_expert_evidence_examples.csv --max_context_seq_length 14 --max_sentence_seq_length 60 --pretrained_embedding_filename /dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin --architecture fullConditional --context_model gru --sentence_model gru --context_rnn_size 100 --sentence_rnn_size 100 --save_dir expert_conditionalstateinput_rnn_rnn_full_lomo_1 --motion_id 1
```

Architecture ConditionalState-RNN-RNN:
```
python TrainRNNBiSequenceClassification.py --task debater --filename ../datasets/task_d-61_study_evidence_examples.csv --max_context_seq_length 14 --max_sentence_seq_length 60 --pretrained_embedding_filename /dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin --architecture conditional --context_model gru --sentence_model gru --context_rnn_size 200 --sentence_rnn_size 200 --batch_size 32 --save_dir study_conditionalstate_rnn_rnn_full_lomo_1 --motion_id 1
```

### Predict script:
Architecture Concat-CNN-CNN:
```
python PredictCNN.py claim_concat_cnn_cnn_full_lomo_1 ../datasets/task_b-88_claim_sentence_examples.csv
```

Architecture ConditionalStateInput-RNN-RNN:
```
python PredictRNNBiSequenceClassification.py expert_conditionalstateinput_rnn_rnn_full_lomo_1 ../datasets/task_c-68_expert_evidence_examples.csv
```

Architecture ConditionalState-RNN-RNN:
```
python PredictRNNBiSequenceClassification.py study_conditionalstate_rnn_rnn_full_lomo_1 ../datasets/task_d-61_study_evidence_examples.csv
```

### Exposing webservice call:
Use LOMOappCNN.py for Architecture Concat-CNN-CNN

Use LOMOappBiSequence.py for Architectures ConditionalState-RNN-RNN and ConditionalStateInput-RNN-RNN
