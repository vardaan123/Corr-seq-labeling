""" prediction function for RNN model
"""

import tensorflow as tf    
import numpy as np
import time
import datetime
import os
import json

from deep.models import RNNSequenceClassification
from deep.dataloaders import PaddedDatasetSequenceClassification 
from deep.utils import Bunch

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["PredictRNNSequenceClassification"]

class PredictRNNSequenceClassification():
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
    model = RNNSequenceClassification(
      num_classes=self.args.num_classes,
      vocab_size=self.args.vocab_size,
      max_seq_length=self.args.max_seq_length,
      embedding_size=self.args.embedding_size,
      rnn_size=self.args.rnn_size,
      num_layers=self.args.num_layers,
      model=self.args.model,
      init_scale_embedding=self.args.init_scale_embedding,
      init_scale=self.args.init_scale,
      train_embedding_matrix=bool(self.args.train_embedding_matrix),
      dynamic_unrolling=bool(self.args.dynamic_unrolling))

    with tf.Graph().as_default():  

      # generate placeholders for the inputs and labels
      self.seq_inputs  = tf.placeholder(tf.int32, shape=(None, self.args.max_seq_length))
      self.seq_lengths = tf.placeholder(tf.int32, shape=(None))
      self.rnn_dropout_keep_prob = tf.placeholder(tf.float32)
      self.output_dropout_keep_prob = tf.placeholder(tf.float32)

      # build a graph that computes predictions from the inference model
      self.predictions_op = model.inference(self.seq_inputs, self.seq_lengths, 
        rnn_dropout_keep_prob = self.rnn_dropout_keep_prob, 
        output_dropout_keep_prob = self.output_dropout_keep_prob)

      # saver to load the model
      saver = tf.train.Saver()

      session_conf = tf.ConfigProto(allow_soft_placement=True)
      self.sess = tf.Session(config=session_conf)

      ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir,'checkpoints'))
      print('Loding the model from [%s]'%(ckpt.model_checkpoint_path))
      saver.restore(self.sess, ckpt.model_checkpoint_path)


  def predict(self,sentences):
    """
    :params:
      sentences: list of strings

    :returns:    
      predictions: predictions, float - [batch_size, num_classes]

    """

    # encode the data
    if self.args.batch_mode == 'pad':
      data = PaddedDatasetSequenceClassification(sentences,None,
        word_to_id = self.word_to_id,
        max_sequence_length = self.args.max_seq_length)

    # run the prediction op in batches
    predictions = np.array([]).reshape(0,self.args.num_classes) 
    while data.epochs_completed < 1:
      batch_seq_inputs,batch_seq_labels,batch_seq_lengths = data.next_batch(self.args.batch_size)
      feed_dict = {
      self.seq_inputs:batch_seq_inputs, 
      self.seq_lengths:batch_seq_lengths, 
      self.rnn_dropout_keep_prob:1.0,
      self.output_dropout_keep_prob:1.0}
      predictions = np.vstack([predictions,self.sess.run(self.predictions_op, feed_dict=feed_dict)])
    
    return predictions 

if __name__ == '__main__':
  sentences = []
  #sentences.append('He argued that violent video games should be banned .')
  #sentences.append('He was a good crow .')
  #sentences.append('The individual freedom of expression is therefore essential to the well-being of society.')
  #sentences.append('In many cases this was given as the least important reason for free trade.')
  sentences.append('Hamdan was to be the first guantanamo detainee tried before a military commission.')
  sentences.append('hayek claimed that a limited democracy might be better than other forms of limited government at protecting liberty but that an unlimited democracy was worse than other forms of unlimited government because its government loses the power even to do what it thinks right if any group on which its majority depends thinks otherwise')

  model = PredictRNNSequenceClassification(model_dir = 'exp_1')
  start_time = time.time()
  predictions = model.predict(sentences)
  duration = time.time() - start_time
  print('Takes %f secs for %d sentences.'%(duration,len(sentences)))
  print predictions
  





