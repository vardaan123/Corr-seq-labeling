
# TED 2 REF.
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabels.py ~/data/seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabels.py ~/data/seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabels.py ~/data/seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabels.py ~/data/seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabels.py ~/data/seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabels.py ~/data/seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabels.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabels.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75

# TED 3 REF.
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5 datasets/TED_3/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_3/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 datasets/TED_3/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_3/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 datasets/TED_3/4classes/data.ndp.np.test.txt datasets/TED_3/4classes/data.ndp.test.lab

# TED 2 ASR.
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5 datasets/TED_2_ASR/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_2_ASR/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 datasets/TED_2_ASR/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_2_ASR/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 datasets/TED_2_ASR/4classes/data.ndp.np.test.txt datasets/TED_2_ASR/4classes/data.ndp.test.lab

# TED 3 ASR.
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps_4cl3_rnn512_gru_bat64_layer2_drop0p5 datasets/TED_3_ASR/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_3_ASR/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/punc7_4cl_rnn256_gru_bat64_layer1 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateSequenceLabelsTest.py ~/data/seq_labelling_logs/caps7_4cl_rnn512_gru_bat64_layer2_drop0p75 datasets/TED_3_ASR/4classes_capt/data.capt.ndp.np.test.txt datasets/TED_3_ASR/4classes_capt/data.capt.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/punc_4cl_rnn512_lstm_hid1_hdim300_linear_bat64_layer1 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn256_lstm_hid1_hdim300_linear_wt0p9_bat64_layer1 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr_4cl3_rnn512_lstm_hid1_hdim300_linear_wt0p75_bat64_layer2_drop1p0 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_gru_hid1_hdim300_relu_wt0p9_bat64_layer2_drop0p5 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab
CUDA_VISIBLE_DEVICES=0 python evaluateCorrelatedSequenceLabelsTest.py ~/data/seq_labelling_logs/corr7_4cl_rnn512_lstm_hid1_hdim300_relu_wt0p75_bat64_layer1_drop0p75 datasets/TED_3_ASR/4classes/data.ndp.np.test.txt datasets/TED_3_ASR/4classes/data.ndp.test.lab