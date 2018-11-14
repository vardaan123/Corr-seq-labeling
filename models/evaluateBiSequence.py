import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

from deep.models import PredictRNNBiSequenceClassification as predict
#from deep.models import PredictCNNSequenceClassification as predict
from deep.utils import word_ids_to_sentence
from deep.metrics import PRCurve, ROCCurve 

from deep.dataloaders.debater_context_data_loaders import load_debater_dataset
train_data, valid_data, test_data = load_debater_dataset(
  filename = sys.argv[1],
  context_free = False,
  train_valid_test_split_ratio = [0.6,0.1,0.3])

model_dir = sys.argv[2]
experiment_name = 'RNN'
model = predict(model_dir = model_dir)

plt.ion()

#-----------------------------------------------------------------------

data = valid_data
label = 'valid'+'-'+experiment_name

start_time = time.time()
predictions = model.predict(data.contexts,data.sentences)
duration = time.time() - start_time
print('[%s] Takes %f secs for %d sentences (%f secs/sentence).'%(label,duration,len(data.sentences),duration/len(data.sentences))) 
y_score = list(predictions[:,1])
y_true = list(data.labels)

PR = PRCurve(y_score, y_true)
PR.plot(PR_label = label,
        line_color = "b", 
        use_existing_figure_number = 'PR curve')
plt.savefig(os.path.join(model.args.save_dir,'valid_prcurve.png'))
ROC = ROCCurve(y_score, y_true)
ROC.plot(ROC_label = label,
  line_color = "b", 
  use_existing_figure_number = 'ROC curve')
plt.savefig(os.path.join(model.args.save_dir,'valid.png'))

#-----------------------------------------------------------------------

with open(os.path.join(model.args.save_dir,'valid_predictions.tsv'),'w') as f:
  f.write('motion_id\tmotion\tsentence\tsentence_as_seen_by_model\ttrue_label\tpredction_score\n')
  for i in xrange(len(y_score)):    
    f.write('%s\t%s\t%s\t%s\t%d\t%f\n'%(
      data.motion_ids[i],
      data.contexts[i],
      data.sentences[i],
      word_ids_to_sentence(list(data.inputs_sentence[i,:]),train_data.id_to_word),
      y_true[i],
      y_score[i]))

#-----------------------------------------------------------------------

data = test_data
label = 'test'+'-'+experiment_name

start_time = time.time()
predictions = model.predict(data.contexts,data.sentences)
duration = time.time() - start_time
print('[%s] Takes %f secs for %d sentences (%f secs/sentence).'%(label,duration,len(data.sentences),duration/len(data.sentences))) 

y_score = list(predictions[:,1])
y_true = list(data.labels)

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

#-----------------------------------------------------------------------

with open(os.path.join(model.args.save_dir,'test_predictions.tsv'),'w') as f:
  f.write('motion_id\tmotion\tsentence\tsentence_as_seen_by_model\ttrue_label\tpredction_score\n')
  for i in xrange(len(y_score)):    
    f.write('%s\t%s\t%s\t%s\t%d\t%f\n'%(
      data.motion_ids[i],
      data.contexts[i],
      data.sentences[i],
      word_ids_to_sentence(list(data.inputs_sentence[i,:]),train_data.id_to_word),
      y_true[i],
      y_score[i]))

#-----------------------------------------------------------------------

