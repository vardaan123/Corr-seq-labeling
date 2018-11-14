import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from deep.sequence_labelling import PredictRNNSequenceLabelling as predict
from deep.utils import word_ids_to_sentence
from deep.metrics import PRCurve, ROCCurve 

from deep.dataloaders.punct_data_loaders import load_single_ted_dataset
model_dir = sys.argv[1]
test_seqfile = sys.argv[2]
test_labfile = sys.argv[3]
outfile = sys.argv[4]
experiment_name = 'RNN'
model = predict(model_dir = model_dir)

test_data = load_single_ted_dataset(test_seqfile,test_labfile,
    clean_sentences = True, vocabulary_min_count = 2)



#plt.ion()

#-----------------------------------------------------------------------
data = test_data
label = 'test'+'-'+experiment_name

start_time = time.time()
predictions = model.predict(data.sentences)
duration = time.time() - start_time
print('[%s] Takes %f secs for %d sentences (%f secs/sentence).'%(label,duration,len(data.sentences),duration/len(data.sentences))) 
print predictions.shape
print data.labels.shape
# y_score = list(predictions[:,1])
# y_true = list(data.labels)
'''
PR = PRCurve(y_score, y_true)
PR.plot(PR_label = label,
        line_color = "k", 
        use_existing_figure_number = 'PR curve')
plt.savefig(os.path.join(model.args.save_dir,'test_prcurve.png'))
ROC = ROCCurve(y_score, y_true)
ROC.plot(ROC_label = label,
  line_color = "k", 
  use_existing_figure_number = 'ROC curve')
plt.savefig(os.path.join(model.args.save_dir,'test.png'))
'''
#-----------------------------------------------------------------------

with open(os.path.join(model.args.save_dir,outfile),'w') as f:
  # f.write('motion_id\tmotion\tsentence\tsentence_as_seen_by_model\ttrue_label\tpredction_score\n')
  for i in xrange(predictions.shape[0]):
    length = data.sequence_lengths[i]
    values = ' '.join([str(v) for v in list(predictions[i,:length])])
    f.write('%s\n'%(values))

#-----------------------------------------------------------------------