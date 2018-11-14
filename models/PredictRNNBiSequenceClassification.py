""" prediction function for RNN model for bisequence
"""

import tensorflow as tf    
import numpy as np
import time
import datetime
import os
import json
import sys
import csv
from deep.models import RNNBiSequenceClassification
from deep.dataloaders import PaddedDatasetBiSequenceClassification 
from deep.utils import Bunch
from deep.utils import build_vocabulary, clean_str
import copy 

__author__  = "Anirban Laha"
__email__   = "anirlaha@in.ibm.com"

__all__ = ["PredictRNNBiSequenceClassification"]

class PredictRNNBiSequenceClassification():
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
    model = RNNBiSequenceClassification(
        num_classes=self.args.num_classes,
        vocab_size=self.args.vocab_size,
        max_context_seq_length=self.args.max_context_seq_length,
        max_sentence_seq_length=self.args.max_sentence_seq_length,
        embedding_size=self.args.embedding_size,
        context_rnn_size=self.args.context_rnn_size,
        sentence_rnn_size=self.args.sentence_rnn_size,
        context_num_layers=self.args.context_num_layers,
        sentence_num_layers=self.args.sentence_num_layers,
        context_model=self.args.context_model,
        sentence_model=self.args.sentence_model,
        init_scale_embedding=self.args.init_scale_embedding,
        init_scale=self.args.init_scale,
        train_embedding_matrix=bool(self.args.train_embedding_matrix),
        architecture = self.args.architecture,
        dynamic_unrolling=bool(self.args.dynamic_unrolling))

    with tf.Graph().as_default():  

      # generate placeholders for the inputs and labels
      self.context_inputs  = tf.placeholder(tf.int32, shape=(None, self.args.max_context_seq_length), name="context_inputs")
      self.sentence_inputs  = tf.placeholder(tf.int32, shape=(None, self.args.max_sentence_seq_length), name="sentence_inputs")
      self.context_lengths = tf.placeholder(tf.int32, shape=(None), name="context_lengths")
      self.sentence_lengths = tf.placeholder(tf.int32, shape=(None), name="sentence_lengths")
      self.context_rnn_dropout_keep_prob = tf.placeholder(tf.float32, name="context_rnn_dropout_keep_prob")
      self.sentence_rnn_dropout_keep_prob = tf.placeholder(tf.float32, name="sentence_rnn_dropout_keep_prob")
      self.output_dropout_keep_prob = tf.placeholder(tf.float32, name="output_dropout_keep_prob")

      # build a graph that computes predictions from the inference model
      self.predictions_op = model.inference(self.context_inputs,
                                  self.context_lengths,
                                  self.sentence_inputs,
                                  self.sentence_lengths, 
                                  self.context_rnn_dropout_keep_prob,
                                  self.sentence_rnn_dropout_keep_prob,
                                  self.output_dropout_keep_prob)

      # saver to load the model
      saver = tf.train.Saver()

      session_conf = tf.ConfigProto(allow_soft_placement=True)
      self.sess = tf.Session(config=session_conf)

      ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir,'checkpoints'))
      modelfilename = ckpt.model_checkpoint_path.strip().split('/')[-1]
      modelfilename = os.path.join(model_dir,'checkpoints',modelfilename)
      print('Loding the model from [%s]'%(modelfilename))
      saver.restore(self.sess, modelfilename)


  def predict(self,contexts,sentences):
    """
    :params:
      contexts: list of strings
      sentences: list of strings

    :returns:    
      predictions: predictions, float - [batch_size, num_classes]

    """

    # encode the data
    if self.args.batch_mode == 'pad':
      data = PaddedDatasetBiSequenceClassification(contexts,sentences,None,
        word_to_id = self.word_to_id,
        max_sequence_length_context = self.args.max_context_seq_length,
        max_sequence_length_sentence = self.args.max_sentence_seq_length)

    # run the prediction op in batches
    predictions = np.array([]).reshape(0,self.args.num_classes) 
    features = np.array([]).reshape(0,self.args.sentence_rnn_size)
    while data.epochs_completed < 1:
        batch_context_inputs, batch_sentence_inputs, batch_labels, batch_context_lengths, batch_sentence_lengths = data.next_batch(self.args.batch_size)
        feed_dict = {self.context_inputs:batch_context_inputs,
                     self.sentence_inputs:batch_sentence_inputs,
                     self.context_lengths:batch_context_lengths,
                     self.sentence_lengths:batch_sentence_lengths,
                     self.context_rnn_dropout_keep_prob:self.args.context_rnn_dropout_keep_prob,
                     self.sentence_rnn_dropout_keep_prob:self.args.sentence_rnn_dropout_keep_prob,
                     self.output_dropout_keep_prob:self.args.output_dropout_keep_prob}
        f,p = self.sess.run(self.predictions_op, feed_dict=feed_dict)
        predictions = np.vstack([predictions,p])
        features = np.vstack([features,f])
    return np.concatenate((features,predictions),axis=1) 
    #return predictions

def readcsv(filename):
    coriginal = []
    soriginal = []
    contexts = []
    sentences = []
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter=',')
        reader.next()
        for line in reader:
            motion_id = int(line[0].strip())
            context = line[1].strip()
            sentence = line[2].strip()
            label = int(line[3].strip())
            coriginal.append(copy.deepcopy(context))
            soriginal.append(copy.deepcopy(sentence))

    	    context = clean_str(context)
            sentence = clean_str(sentence)
            contexts.append(context)
            sentences.append(sentence)
    return contexts,sentences,coriginal,soriginal



if __name__ == '__main__':
  # sentences = []
  # sentences.append('Hamdan was to be the first guantanamo detainee tried before a military commission.')
  # sentences.append('hayek claimed that a limited democracy might be better than other forms of limited government at protecting liberty but that an unlimited democracy was worse than other forms of unlimited government because its government loses the power even to do what it thinks right if any group on which its majority depends thinks otherwise')

  # contexts = []
  # contexts.append('Military detention')
  # contexts.append('Democracy should be limited')
  from deep.dataloaders.debater_context_data_loaders import load_debater_dataset

  
  model = PredictRNNBiSequenceClassification(model_dir = sys.argv[1])
  # train_data, valid_data, test_data = load_debater_dataset(
    # filename = model.args.filename,
    # context_free = False,
    # clean_sentences = True,
    # vocabulary_min_count = model.args.vocab_min_count,
    # mode = model.args.batch_mode,
    # max_sequence_length_context = model.args.max_context_seq_length,
    # max_sequence_length_sentence = model.args.max_sentence_seq_length,        
    # train_valid_test_split_ratio = [0.95,0.049,0.001])
  contexts,sentences,coriginal,soriginal = readcsv(sys.argv[2])
  start_time = time.time()
  predictions = model.predict(contexts, sentences)
  #predictions = model.predict(test_data.contexts,test_data.sentences)
  duration = time.time() - start_time
  print('Takes %f secs for test predictions.'%(duration))
  #predictions = np.append(predictions,test_data.labels.reshape(test_data.labels.shape[0],1),axis=1)
  #np.savetxt(os.path.join(model.args.save_dir, 'predictions', 'test.txt'),predictions,fmt='%f',delimiter=',')
  with open(os.path.join(model.args.save_dir, 'predictions', 'dump.txt'),'w') as f:
        for i in range(len(coriginal)):
            ctxt = coriginal[i]
            sent = soriginal[i]
            feat = [str(sr) for sr in list(predictions[i])]
            f.write('%s\t%s\t%s\n'%(ctxt,sent,'\t'.join(feat)))
  f.close()
