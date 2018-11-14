""" prediction function for using RNN for sequence labelling
"""

import tensorflow as tf    
import numpy as np
import time
import datetime
import os
import json

from deep.sequence_labelling import RNNSequenceLabelling
from deep.sequence_labelling import BiRNNSequenceLabelling
from deep.sequence_labelling import CorrelatedSequenceLabelling
from deep.dataloaders import FullCorrelatedDatasetSequenceLabelling 
from deep.utils import Bunch

__author__  = "Vardaan Pahuja,Anirban Laha"
__email__   = "vapahuja@in.ibm.com,anirlaha@in.ibm.com"

__all__ = ["PredictCorrelatedSequenceLabelling"]

class PredictCorrelatedSequenceLabelling():
  """ prediction 
  """
  def __init__(self,model_dir):
    """ load the trained tensorflow model

    :params:
      model_dir: string
        director where the trained tensorflow model is saved
    """

    # load the hyperparameters
    with open(os.path.join(model_dir,'args.json'),'r') as f:
      self.args = Bunch(json.loads(f.read()))

    # load the vocabulary 
    with open(os.path.join(model_dir,'word_to_id.json'),'r') as f:
      self.word_to_id = json.loads(f.read())

    # dummy value for the initial embedding matrix
    self.initial_embedding_matrix = np.zeros(shape=(self.args.vocab_size,self.args.embedding_size),dtype=np.float32)

    # initialize the model

    model = CorrelatedSequenceLabelling(
        num_classes=self.args.num_classes,
        vocab_size=self.args.vocab_size,
        max_seq_length=self.args.max_seq_length,
        bidirectional=bool(self.args.bidirectional),
        embedding_size=self.args.embedding_size,
        rnn_size=self.args.rnn_size,
        num_layers=self.args.num_layers,
        model=self.args.model,
        use_hidden_layer=bool(self.args.use_hidden_layer),
        hidden_dimension=self.args.hidden_dimension,
        hidden_activation=self.args.hidden_activation,
        num_tasks=self.args.num_tasks,
        task_weights=[float(x) for x in self.args.task_loss_weight.split(':')],
        init_scale_embedding=self.args.init_scale_embedding,
        init_scale=self.args.init_scale,
        train_embedding_matrix=bool(self.args.train_embedding_matrix),
        dynamic_unrolling=bool(self.args.dynamic_unrolling))

    with tf.Graph().as_default():  

      # generate placeholders for the inputs and labels
      self.seq_inputs  = tf.placeholder(tf.int32, shape=(None, None))
      self.seq_lengths = tf.placeholder(tf.int32, shape=(None))
      self.rnn_dropout_keep_prob = tf.placeholder(tf.float32)
      self.output_dropout_keep_prob = tf.placeholder(tf.float32)

      # build a graph that computes predictions from the inference model
      self.logits_op = model.inference(self.seq_inputs, self.seq_lengths, 
        rnn_dropout_keep_prob = self.rnn_dropout_keep_prob, 
        output_dropout_keep_prob = self.output_dropout_keep_prob)
      self.predictions_op = model.prediction(self.logits_op)

      # saver to load the model
      saver = tf.train.Saver()

      session_conf = tf.ConfigProto(allow_soft_placement=True)
      self.sess = tf.Session(config=session_conf)

      ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir,'checkpoints'))
      modelfilename = ckpt.model_checkpoint_path.strip().split('/')[-1]
      modelfilename = os.path.join(model_dir,'checkpoints',modelfilename)
      print('Loading the model from [%s]'%(modelfilename))
      saver.restore(self.sess, modelfilename)
      

  def predict(self,sentences):
    """
    :params:
      sentences: list of strings

    :returns:    
      predictions: predictions, float - [len(sentences), data.max_sequence_length, num_classes]

    """

    # encode the data
    if self.args.batch_mode == 'full':
      data = FullCorrelatedDatasetSequenceLabelling(sentences,None,self.args.num_tasks,
        word_to_id = self.word_to_id,
        max_sequence_length = None)
    elif self.args.batch_mode == 'truncate':
      data = FullCorrelatedDatasetSequenceLabelling(sentences,None,self.args.num_tasks,
        word_to_id = self.word_to_id,
        max_sequence_length = self.args.max_seq_length)

    # run the prediction op in batches

    predictions_list = []

    for i in range(self.args.num_tasks):
      predictions_list.append(np.array([]).reshape(0,data.max_sequence_length,self.args.num_classes[i]))

    while data.epochs_completed < 1:
      batch_seq_inputs,batch_seq_labels,batch_seq_lengths = data.next_batch(self.args.batch_size)
      feed_dict = {
      self.seq_inputs:batch_seq_inputs, 
      self.seq_lengths:batch_seq_lengths, 
      self.rnn_dropout_keep_prob:1.0,
      self.output_dropout_keep_prob:1.0}
      
      output = self.sess.run(self.predictions_op, feed_dict=feed_dict)
      batch_max_seq_length = np.max(batch_seq_lengths)

      for task_id in range(self.args.num_tasks):
        current_pred = np.zeros([batch_seq_lengths.shape[0],data.max_sequence_length, self.args.num_classes[task_id]], dtype=np.float64)
        output_task = output[task_id]
        for id,length in enumerate(batch_seq_lengths):
              current_pred[id,:batch_max_seq_length] = output_task[id]
        predictions_list[task_id] = np.concatenate([predictions_list[task_id],current_pred],axis=0)

    pred_labels_list = []

    for task_id in range(self.args.num_tasks):
      pred_labels = np.zeros([len(sentences), data.max_sequence_length], dtype=np.int32)
      predictions = predictions_list[task_id]
      for id,tokens in enumerate(data.input_sequences):
          for l in range(len(tokens)):
              pred_labels[id,l] = np.argmax(predictions[id,l])
      pred_labels_list.append(pred_labels)

    return pred_labels_list


    

  
    
    

if __name__ == '__main__':
  sentences = []
  sentences.append("in china we have 500 million internet users that 's the biggest population of UNK internet users in the whole world so even though china 's is a totally censored internet but still chinese internet society is really booming how to make it ? it 's simple you have google we have baidu you have twitter we have UNK")
  sentences.append("for example mubarak he shut down the internet he wanted to prevent the UNK [ from criticizing ] him but once UNK ca n't go online they go in the street")
  sentences.append("now a block of limestone in itself is n't particularly that interesting it looks beautiful but imagine what the properties of this limestone block might be if the surfaces were actually in conversation with the atmosphere maybe they could extract carbon dioxide would it give this block of limestone new properties ? well most likely it would")

  model = PredictCorrelatedSequenceLabelling(model_dir = 'corr_hidden_both')
  start_time = time.time()
  predictions = model.predict(sentences)
  duration = time.time() - start_time
  print('Takes %f secs for %d sentences.'%(duration,len(sentences)))

  for pred_labels in predictions:
    print predictions
  
