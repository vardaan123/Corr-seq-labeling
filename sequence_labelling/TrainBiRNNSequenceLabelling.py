""" training script for BiRNNSequenceLabelling
<TODO> : Integrate it as one option in TrainRNNSequenceLabelling.py
<TODO> : Make the summary computation for validation efficient (currently commented)
"""

import tensorflow as tf    
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime
import os
import json
import math

from deep.utils  import change_default_model_in_checkpoint
from deep.sequence_labelling import BiRNNSequenceLabelling
from deep.models import PretrainedEmbeddings

__author__  = "Vardaan Pahuja"
__email__   = "vapahuja@in.ibm.com"

__all__ = ["TrainBiRNNSequenceLabelling"]

def TrainBiRNNSequenceLabelling(args,train_data,valid_data):
  """ training function
  """

  # ==================================================
  # PRE-TRAINED EMBEDDINGS
  # ==================================================

  if args.use_pretrained_embedding_matrix:
    print('----------------------------loading word2vec embeddings')
    embeddings = PretrainedEmbeddings(args.pretrained_embedding_filename)
    initial_embedding_matrix = embeddings.load_embedding_matrix(train_data.word_to_id)
    args.embedding_size = embeddings.embedding_size

  # ==================================================
  # SAVE HYPERPARAMETERS AND VOCABULARY
  # ==================================================

  print('----------------------------HYPERPARAMETES')
  for arg in vars(args):
    print('%s = %s'%(arg, getattr(args, arg)))

  # save the args and the vocabulary

  if not os.path.exists(args.save_dir): 
    os.makedirs(args.save_dir)  

  with open(os.path.join(args.save_dir,'args.json'),'w') as f:
    f.write(json.dumps(vars(args),indent=1))

  with open(os.path.join(args.save_dir,'word_to_id.json'),'w') as f:
    f.write(json.dumps(train_data.word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'id_to_word.json'),'w') as f:
    f.write(json.dumps(train_data.id_to_word,indent=1))

  # ==================================================
  # TRAINING
  # ==================================================

  model = BiRNNSequenceLabelling(
    num_classes=args.num_classes,
    vocab_size=args.vocab_size,
    max_seq_length=args.max_seq_length,
    bidirectional=bool(args.bidirectional),
    embedding_size=args.embedding_size,
    rnn_size=args.rnn_size,
    num_layers=args.num_layers,
    model=args.model,
    init_scale_embedding=args.init_scale_embedding,
    init_scale=args.init_scale,
    train_embedding_matrix=bool(args.train_embedding_matrix),
    dynamic_unrolling=bool(args.dynamic_unrolling))

  print('----------------------------TRAINING')

  total_start_time = time.time()   

  with tf.Graph().as_default():  

    # generate placeholders for the inputs, seq_lengths, labels, and dropout
    seq_inputs  = tf.placeholder(tf.int32, shape=(None, None), name="seq_inputs")
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name="seq_lengths")
    labels = tf.placeholder(tf.int32, shape=(None, None), name="labels")
    rnn_dropout_keep_prob = tf.placeholder(tf.float32, name="rnn_dropout_keep_prob")
    output_dropout_keep_prob = tf.placeholder(tf.float32, name="output_dropout_keep_prob")

    # build a graph that computes predictions from the inference model
    # logits or predictions??
    logits_op = model.inference(seq_inputs, seq_lengths, 
      rnn_dropout_keep_prob = rnn_dropout_keep_prob, 
      output_dropout_keep_prob = output_dropout_keep_prob)
    #predictions_op = tf.nn.softmax(logits_op)

    # add to the graph the ops for loss calculation
    if not bool(args.use_cost_sensitive_loss):
        loss_op = model.loss(logits_op, seq_lengths, labels)
    # else:
        # loss_op = model.loss_pos_weighted(logits_op, seq_lengths, labels)

    # add to the graph the ops that calculate and apply gradients
    train_op = model.training(loss_op,
      optimizer = args.optimizer,
      learning_rate = args.learning_rate)

    # add the op to compare the predictions to the labels during evaluation
    accuracy_op = model.evaluation(logits_op, labels)

    # summaries for loss and accuracy
    loss_summary = tf.scalar_summary('loss', loss_op)
    accuracy_summary  = tf.scalar_summary('accuracy', accuracy_op)

    train_summary_op = tf.merge_summary([loss_summary, accuracy_summary])
    valid_summary_op = tf.merge_summary([loss_summary, accuracy_summary])

    # directory to dump the summaries

    train_summary_dir = os.path.join(args.save_dir, 'summaries', 'train')
    if not os.path.exists(train_summary_dir): 
      os.makedirs(train_summary_dir)

    valid_summary_dir = os.path.join(args.save_dir, 'summaries', 'valid')
    if not os.path.exists(valid_summary_dir): 
      os.makedirs(valid_summary_dir)

    # directory to dump the intermediate models
    checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')  

    # location to dump the predictions for validation
    if not os.path.exists(os.path.join(args.save_dir, 'predictions')):
      os.makedirs(os.path.join(args.save_dir, 'predictions'))

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=max(5,args.early_stopping_batch_window+2))

    # evaluation function
    def evaluate_model(sess,
      data_set,
      seq_inputs,
      seq_lengths,
      labels,
      accuracy_op,
      loss_op,
      batch_size):
      """ evaluate on the entire data to track some metrics
      """
      batch_accuracy = 0.0 
      batch_loss = 0.0
      steps_per_epoch = data_set.num_examples // batch_size + 1
      num_predictions = 0
      for step in xrange(steps_per_epoch):
        batch_seq_inputs,batch_labels,batch_seq_lengths = data_set.next_batch(batch_size)
        feed_dict = {seq_inputs:batch_seq_inputs,
        labels:batch_labels,
        seq_lengths:batch_seq_lengths,
        rnn_dropout_keep_prob:1.0,
        output_dropout_keep_prob:1.0}
        batch_accuracy += sess.run(accuracy_op,feed_dict=feed_dict)*np.sum(batch_seq_lengths)
        batch_loss += sess.run(loss_op,feed_dict=feed_dict)*np.shape(batch_seq_lengths)[0]
        num_predictions += np.sum(batch_seq_lengths)
      accuracy = batch_accuracy / num_predictions
      loss = batch_loss / data_set.num_examples
      
      # print 'accuracy:' + str(accuracy)
      # print 'loss:' + str(loss)
      return accuracy,loss

    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf) as sess:
      # run the op to initialize the variables
      init = tf.initialize_all_variables()
      sess.run(init)    

      # use the pre-trained word2vec embeddings
      if args.use_pretrained_embedding_matrix:
        sess.run(model.embedding_matrix.assign(initial_embedding_matrix))
     
      print('Trainable Variables')
      print '\n'.join([v.name for v in tf.trainable_variables()])

      # instantiate a SummaryWriter to output summaries and the graph
      train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph_def=sess.graph_def)
      valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, graph_def=sess.graph_def)

      #for step in xrange(1,args.max_steps+1):

      step = 0
      previous_epoch = 0
      performance = {}
      performance['valid_loss'] = []
      performance['train_loss'] = []
      performance['valid_accuracy'] = []
      performance['train_accuracy'] = []    
      while train_data.epochs_completed <= args.max_epochs:

        step += 1
       
        start_time = time.time()      
        batch_seq_inputs,batch_labels,batch_seq_lengths = train_data.next_batch(args.batch_size)
        feed_dict = {
        seq_inputs:batch_seq_inputs, 
        labels:batch_labels, 
        seq_lengths:batch_seq_lengths, 
        rnn_dropout_keep_prob:args.rnn_dropout_keep_prob,
        output_dropout_keep_prob:args.output_dropout_keep_prob}
        _, loss, accuracy = sess.run([train_op, loss_op, accuracy_op],feed_dict=feed_dict)
        duration = time.time() - start_time

        # print an overview 
        if step % args.print_every == 0:
          print('epoch %d batch %d: loss = %.3f accuracy = %.2f (%.3f secs)' % (train_data.epochs_completed+1,
            step,
            loss,
            accuracy,
            duration))

        # write the summaries
        if step % args.summary_every == 0:

          summaries = sess.run(train_summary_op, feed_dict=feed_dict)
          train_summary_writer.add_summary(summaries, step)

          # feed_dict = {
          # seq_inputs:valid_data.inputs, 
          # labels:valid_data.labels, 
          # seq_lengths:valid_data.sequence_lengths, 
          # rnn_dropout_keep_prob:args.rnn_dropout_keep_prob,
          # output_dropout_keep_prob:args.output_dropout_keep_prob}

          # summaries = sess.run(valid_summary_op, feed_dict=feed_dict)
          # valid_summary_writer.add_summary(summaries, step)
              
        # if epoch completed
        if train_data.epochs_completed > previous_epoch:

          previous_epoch = train_data.epochs_completed

          # evaluate the model on the entire data    
          if train_data.epochs_completed % args.evaluate_every_epochs == 0 or train_data.epochs_completed == args.max_epochs-1:
            accuracy,loss = evaluate_model(sess,train_data,seq_inputs,seq_lengths,labels,accuracy_op,loss_op,args.batch_size)                
            print('----------------------------------train accuracy : %0.03f loss : %0.03f' %(accuracy,loss))          
            performance['train_loss'].append(loss)
            performance['train_accuracy'].append(accuracy)
            train_data.reset_batch(epochs_completed=previous_epoch)

            accuracy,loss = evaluate_model(sess,valid_data,seq_inputs,seq_lengths,labels,accuracy_op,loss_op,args.batch_size)
            print('----------------------------------valid accuracy : %0.03f loss : %0.03f' %(accuracy,loss))
            performance['valid_loss'].append(loss)
            performance['valid_accuracy'].append(accuracy)
            valid_data.reset_batch()

          # save a checkpoint 
          if train_data.epochs_completed % args.save_every_epochs == 0 or train_data.epochs_completed == args.max_epochs-1: 
            path = saver.save(sess, checkpoint_prefix, global_step=train_data.epochs_completed)
            print("Saved model checkpoint to {}".format(path))

        # early stopping 
        if args.early_stopping:    
          if train_data.epochs_completed > args.early_stopping_batch_window:
            current_loss  = performance['valid_loss'][-1]
            previous_loss = performance['valid_loss'][-1-args.early_stopping_batch_window]
            if (current_loss-previous_loss)/(previous_loss) > args.early_stopping_threshold or math.isnan(current_loss):
              print('----------------------------------EARLY STOPPING')   
              checkpoint_file_path = os.path.join(checkpoint_dir,'checkpoint')
              backtrack = train_data.epochs_completed-args.early_stopping_batch_window+1
              model_file_path = '%s-%s'%(checkpoint_prefix,backtrack)
              print("Backtracking model checkpoint to {}".format(model_file_path))            
              change_default_model_in_checkpoint(checkpoint_file_path,model_file_path)
              break

  total_duration = time.time() - total_start_time  
  performance['time_taken'] = total_duration
  print('Total time taken = %1.2f minutes'%(total_duration/60.0))

  # ==================================================
  # SAVE THE CONVERGENCE PROGRESS
  # ==================================================

  with open(os.path.join(args.save_dir,'performance.json'),'w') as f:
    f.write(json.dumps(performance,indent=1))        

  try:
      plt.ion()
      plt.figure()        
      epochs = range(1,len(performance['train_loss'])+1)
      plt.plot(epochs,performance['train_loss'],'r.-',label='train')
      plt.plot(epochs,performance['valid_loss'],'k.-',label='valid')
      plt.xlabel('epoch')
      plt.ylabel('loss')
      plt.legend(loc='best')
      #plt.show()
      plt.savefig(os.path.join(args.save_dir,'loss.jpeg'))
      plt.close()
  except:
      print 'Error occurred in plotting... Please refer performance.json to get the values'

if __name__ == '__main__':
  """ example usage
  """

  # ==================================================
  # PARAMETERS
  # ==================================================

  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--bidirectional', type=int, default=0, help='use bidirectional rnn or not') 
  parser.add_argument('--embedding_size', type=int, default=64, help='size of the word embeddings') # 100,200 32,64,128
  parser.add_argument('--rnn_size', type=int, default=64, help='size of RNN hidden state') # 100
  parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN (default: 1)') 
  parser.add_argument('--model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: basic_lstm)')
  parser.add_argument('--rnn_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability when using multiple RNNs')
  parser.add_argument('--output_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability for the output (i.e. before the softmax layer)')
  parser.add_argument('--init_scale_embedding', type=float, default=1.0, help='random uniform initialization in the range [-init_scale_embedding,init_scale_embedding] for the embeddign layer')
  parser.add_argument('--init_scale', type=float, default=0.1, help='random uniform initialization in the range [-init_scale,init_scale]')
  parser.add_argument('--train_embedding_matrix', type=int, default=1, help='if 0 does not train the embedding matrix and keeps it fixed')
  parser.add_argument('--use_pretrained_embedding_matrix', type=int, default=1, help='if 1 use the pre-trained word2vec for initializing the embedding matrix')
  parser.add_argument('--pretrained_embedding_filename', type=str, default='../resources/GoogleNews-vectors-negative300.bin', help='full path to the .bin file containing the pre-trained word vectors')              
  parser.add_argument('--dynamic_unrolling', type=int, default=1, help='if 1 do dynamic calculations when unrolling the RNN')
  parser.add_argument('--use_cost_sensitive_loss', type=int, default=0, help='if 1 use_cost_sensitive_loss only if binary classification')
   
  # Training parameters
  parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='cfree_claim_3', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after this many batches')
  parser.add_argument('--summary_every', type=int, default=10, help='dump summaries for tensorboard after this many batches')
  parser.add_argument('--save_every_epochs', type=int, default=1, help='save the model after this many epochs')
  parser.add_argument('--evaluate_every_epochs', type=int, default=1, help='evaluate the model on the entire data after this many epochs')

  parser.add_argument('--early_stopping', type=int, default=1, help='if 1 enables early stopping based on loss on the validation split')
  parser.add_argument('--early_stopping_batch_window', type=int, default=5, help='early stop if the validation loss is greater than the loss from these many previous steps')
  parser.add_argument('--early_stopping_threshold', type=float, default=0.05, help='threshold for early stopping')

  # Task specific
  parser.add_argument('--trainFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--trainLabelFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--testFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--testLabelFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--devFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--devLabelFilename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--vocab_min_count', type=int, default=5, help='keep words whose count is >= vocab_min_count') # 5
  parser.add_argument('--batch_mode', type=str, default='full', help='pad, bucket : how to handle variable length sequences in a batch')
  parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
  parser.add_argument('--vocab_size', type=int, default=None, help='size of the vocabulary')
  parser.add_argument('--max_seq_length', type=int, default=50, help='maximum sequence length allowed') # 50

  args = parser.parse_args()

  # ==================================================
  # LOAD THE DATASET HERE
  # ==================================================
  '''
  from deep.dataloaders.claim_data_loaders import load_context_free_claim_sentence_dataset

  train_data, valid_data, test_data = load_context_free_claim_sentence_dataset(
    filename = args.filename,
    clean_sentences = True,
    vocabulary_min_count = args.vocab_min_count,
    mode = args.batch_mode,
    max_sequence_length = args.max_seq_length,
    train_valid_test_split_ratio = [0.5,0.3,0.2])

  args.num_classes = train_data.num_classes
  args.vocab_size = train_data.vocabulary_size
  args.max_seq_length = train_data.max_sequence_length
  '''
  from deep.dataloaders.punct_data_loaders import load_ted_dataset

  train_data, valid_data, test_data = load_ted_dataset(args.trainFilename,
    args.devFilename,
    args.testFilename,
    args.trainLabelFilename,
    args.devLabelFilename,
    args.testLabelFilename,
    mode=args.batch_mode,
    max_seq_length=args.max_seq_length,
    #clean_sentences = False,
    clean_sentences = True,
    vocabulary_min_count = args.vocab_min_count)

  args.num_classes = train_data.num_classes
  args.vocab_size = train_data.vocabulary_size
  args.max_seq_length = train_data.max_sequence_length

  # ==================================================
  # TRAINING
  # ==================================================

  TrainBiRNNSequenceLabelling(args,train_data,valid_data)
