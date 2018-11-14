""" training script for RNNBiSequenceClassification
"""

import tensorflow as tf    
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime
import os
import json
import traceback
from shutil import copyfile
import math

from deep.utils  import change_default_model_in_checkpoint
from deep.models import RNNBiSequenceClassification
from deep.models import PretrainedEmbeddings
from deep.metrics import PRCurve,ROCCurve

__author__  = "Anirban Laha"
__email__   = "anirlaha@in.ibm.com"

__all__ = ["TrainRNNBiSequenceClassification"]

def TrainRNNBiSequenceClassification(args,train_data,valid_data):
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

  model = RNNBiSequenceClassification(
    num_classes=args.num_classes,
    vocab_size=args.vocab_size,
    max_context_seq_length=args.max_context_seq_length,
    max_sentence_seq_length=args.max_sentence_seq_length,
    embedding_size=args.embedding_size,
    context_rnn_size=args.context_rnn_size,
    sentence_rnn_size=args.sentence_rnn_size,
    context_num_layers=args.context_num_layers,
    sentence_num_layers=args.sentence_num_layers,
    context_model=args.context_model,
    sentence_model=args.sentence_model,
    init_scale_embedding=args.init_scale_embedding,
    init_scale=args.init_scale,
    train_embedding_matrix=bool(args.train_embedding_matrix),
    architecture = args.architecture,
    dynamic_unrolling=bool(args.dynamic_unrolling),
    class_ratios=[1-args.positive_proportion,args.positive_proportion])

  print('----------------------------TRAINING')

  total_start_time = time.time()   

  with tf.Graph().as_default():  

    # generate placeholders for the inputs, seq_lengths, labels, and dropout
    context_inputs  = tf.placeholder(tf.int32, shape=(None, args.max_context_seq_length), name="context_inputs")
    sentence_inputs  = tf.placeholder(tf.int32, shape=(None, args.max_sentence_seq_length), name="sentence_inputs")
    context_lengths = tf.placeholder(tf.int32, shape=(None), name="context_lengths")
    sentence_lengths = tf.placeholder(tf.int32, shape=(None), name="sentence_lengths")
    labels = tf.placeholder(tf.int32, shape=(None), name="labels")
    context_rnn_dropout_keep_prob = tf.placeholder(tf.float32, name="context_rnn_dropout_keep_prob")
    sentence_rnn_dropout_keep_prob = tf.placeholder(tf.float32, name="sentence_rnn_dropout_keep_prob")
    output_dropout_keep_prob = tf.placeholder(tf.float32, name="output_dropout_keep_prob")

    # build a graph that computes predictions from the inference model
    logits_op = model.inference(context_inputs,
                                  context_lengths,
                                  sentence_inputs,
                                  sentence_lengths,
                                  context_rnn_dropout_keep_prob,
                                  sentence_rnn_dropout_keep_prob,
                                  output_dropout_keep_prob,
                                  return_logit=True)
    predictions_op = tf.nn.softmax(logits_op)

    # add to the graph the ops for loss calculation
    if args.num_classes > 2 or not bool(args.use_cost_sensitive_loss):
        loss_op = model.loss(predictions_op, labels)
    else:
        loss_op = model.loss_pos_weighted(logits_op, labels)
        print 'Cost-sensitive loss applied'

    # add to the graph the ops that calculate and apply gradients
    train_op = model.training(loss_op,
      optimizer = args.optimizer,
      learning_rate = args.learning_rate)

    # add the op to compare the predictions to the labels during evaluation
    accuracy_op = model.evaluation(predictions_op, labels)

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
    valid_location = os.path.join(args.save_dir, 'predictions', 'valid.txt')
    if not os.path.exists(os.path.join(args.save_dir, 'predictions')):
      os.makedirs(os.path.join(args.save_dir, 'predictions'))

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=max(5,args.early_stopping_batch_window+2))

    # evaluation function
    def evaluate_model(sess,
      data_set,
      context_inputs,
      context_lengths,
      sentence_inputs,
      sentence_lengths,
      labels,
      accuracy_op,
      loss_op,
      batch_size):
      """ evaluate on the entire data to track some metrics
      """
      batch_accuracy = 0.0 
      batch_loss = 0.0
      steps_per_epoch = data_set.num_examples // batch_size
      num_examples = steps_per_epoch * batch_size
      for step in xrange(steps_per_epoch):
        batch_context_inputs, batch_sentence_inputs, batch_labels, batch_context_lengths, batch_sentence_lengths = data_set.next_batch(batch_size)
        feed_dict = {context_inputs:batch_context_inputs,
                     sentence_inputs:batch_sentence_inputs,
                     labels:batch_labels,
                     context_lengths:batch_context_lengths,
                     sentence_lengths:batch_sentence_lengths,
                     context_rnn_dropout_keep_prob:1.0,
                     sentence_rnn_dropout_keep_prob:1.0,
                     output_dropout_keep_prob:1.0}
        batch_accuracy += sess.run(accuracy_op,feed_dict=feed_dict)*batch_size
        batch_loss += sess.run(loss_op,feed_dict=feed_dict)*batch_size
      accuracy = batch_accuracy / num_examples
      loss = batch_loss / num_examples
      
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
      while train_data.epochs_completed < args.max_epochs:

        step += 1
       
        start_time = time.time()
        batch_context_inputs, batch_sentence_inputs, batch_labels, batch_context_lengths, batch_sentence_lengths = train_data.next_batch(args.batch_size)
        feed_dict = {context_inputs:batch_context_inputs,
                     sentence_inputs:batch_sentence_inputs,
                     labels:batch_labels,
                     context_lengths:batch_context_lengths,
                     sentence_lengths:batch_sentence_lengths,
                     context_rnn_dropout_keep_prob:args.context_rnn_dropout_keep_prob,
                     sentence_rnn_dropout_keep_prob:args.sentence_rnn_dropout_keep_prob,
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

          feed_dict = {context_inputs:valid_data.inputs_context,
                     sentence_inputs:valid_data.inputs_sentence,
                     labels:valid_data.labels,
                     context_lengths:valid_data.seq_lengths_context,
                     sentence_lengths:valid_data.seq_lengths_sentence,
                     context_rnn_dropout_keep_prob:args.context_rnn_dropout_keep_prob,
                     sentence_rnn_dropout_keep_prob:args.sentence_rnn_dropout_keep_prob,
                     output_dropout_keep_prob:args.output_dropout_keep_prob}

          summaries = sess.run(valid_summary_op, feed_dict=feed_dict)
          valid_summary_writer.add_summary(summaries, step)
              
        # if epoch completed
        if train_data.epochs_completed > previous_epoch:

          previous_epoch = train_data.epochs_completed

          # evaluate the model on the entire data    
          if train_data.epochs_completed % args.evaluate_every_epochs == 0 or train_data.epochs_completed == args.max_epochs-1:
            accuracy,loss = evaluate_model(sess,train_data,context_inputs,context_lengths,sentence_inputs,sentence_lengths,labels,accuracy_op,loss_op,args.batch_size)                
            print('----------------------------------train accuracy : %0.03f loss : %0.03f' %(accuracy,loss))          
            performance['train_loss'].append(loss)
            performance['train_accuracy'].append(accuracy)
            train_data.reset_batch(epochs_completed=previous_epoch)

            accuracy,loss = evaluate_model(sess,valid_data,context_inputs,context_lengths,sentence_inputs,sentence_lengths,labels,accuracy_op,loss_op,args.batch_size)
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
  parser.add_argument('--embedding_size', type=int, default=64, help='size of the word embeddings') # 100,200 32,64,128
  parser.add_argument('--context_rnn_size', type=int, default=64, help='size of context RNN hidden state') # 100
  parser.add_argument('--sentence_rnn_size', type=int, default=64, help='size of sentence RNN hidden state') # 100
  parser.add_argument('--context_num_layers', type=int, default=1, help='number of layers in the context RNN (default: 1)') 
  parser.add_argument('--sentence_num_layers', type=int, default=1, help='number of layers in the sentence RNN (default: 1)') 
  parser.add_argument('--context_model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: basic_lstm)')
  parser.add_argument('--sentence_model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: basic_lstm)')
  parser.add_argument('--context_rnn_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability when using multiple RNNs for context')
  parser.add_argument('--sentence_rnn_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability when using multiple RNNs for sentence')
  parser.add_argument('--output_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability for the output (i.e. before the softmax layer)')
  parser.add_argument('--init_scale_embedding', type=float, default=1.0, help='random uniform initialization in the range [-init_scale_embedding,init_scale_embedding] for the embeddign layer')
  parser.add_argument('--init_scale', type=float, default=0.1, help='random uniform initialization in the range [-init_scale,init_scale]')
  parser.add_argument('--train_embedding_matrix', type=int, default=1, help='if 0 does not train the embedding matrix and keeps it fixed')
  parser.add_argument('--use_pretrained_embedding_matrix', type=int, default=1, help='if 1 use the pre-trained word2vec for initializing the embedding matrix')
  parser.add_argument('--pretrained_embedding_filename', type=str, default='../resources/GoogleNews-vectors-negative300.bin', help='full path to the .bin file containing the pre-trained word vectors')              
  parser.add_argument('--dynamic_unrolling', type=int, default=1, help='if 1 do dynamic calculations when unrolling the RNN')
  parser.add_argument('--use_cost_sensitive_loss', type=int, default=0, help='if 1 use_cost_sensitive_loss only if binary classification')
  parser.add_argument('--positive_proportion', type=int, default=0.03, help='used only if cost_sensitive_loss is enabled')
   
  # Training parameters
  parser.add_argument('--architecture', type=str, default='concat', help='choose the architecture variant for combining two sequences')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='exp_1', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after this many batches')
  parser.add_argument('--summary_every', type=int, default=10, help='dump summaries for tensorboard after this many batches')
  parser.add_argument('--save_every_epochs', type=int, default=1, help='save the model after this many epochs')
  parser.add_argument('--evaluate_every_epochs', type=int, default=1, help='evaluate the model on the entire data after this many epochs')

  parser.add_argument('--early_stopping', type=int, default=1, help='if 1 enables early stopping based on loss on the validation split')
  parser.add_argument('--early_stopping_batch_window', type=int, default=5, help='early stop if the validation loss is greater than the loss from these many previous steps')
  parser.add_argument('--early_stopping_threshold', type=float, default=0.05, help='threshold for early stopping')

  # Task specific
  parser.add_argument('--task', type=str, default='debater', help='name of the task being considered. dataset will be chosen accordingly.')
  parser.add_argument('--filename', type=str, default=None, help='name of the file which contains debater dataset')
  parser.add_argument('--motion_id', type=int, default=None, help='test motion_id if you consider to run in LOMO mode')
  parser.add_argument('--vocab_min_count', type=int, default=5, help='keep words whose count is >= vocab_min_count') # 5
  parser.add_argument('--batch_mode', type=str, default='pad', help='pad, bucket : how to handle variable length sequences in a batch')
  parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
  parser.add_argument('--vocab_size', type=int, default=None, help='size of the vocabulary')
  parser.add_argument('--max_context_seq_length', type=int, default=50, help='maximum sequence length allowed for context') # 50
  parser.add_argument('--max_sentence_seq_length', type=int, default=50, help='maximum sequence length allowed for sentence') # 50

  args = parser.parse_args()

  # ==================================================
  # LOAD THE DATASET HERE
  # ==================================================

  from deep.dataloaders.public_data_loaders import load_public_dataset, load_public_dataset_split_train_valid
  from deep.dataloaders.debater_context_data_loaders import load_debater_dataset, load_debater_dataset_LOMO_mode

  # For the textual entailment dataset
  if args.task == 'textual_entailment':
	  train_data, valid_data, test_data = load_public_dataset(
	    filename = '../datasets/snli_1.0_textual_entailment/snli_1.0/trainfile.txt',
	    validFilename = '../datasets/snli_1.0_textual_entailment/snli_1.0/devfile.txt',
	    testFilename = '../datasets/snli_1.0_textual_entailment/snli_1.0/testfile.txt',
	    clean_sentences = True,
	    vocabulary_min_count = args.vocab_min_count,
	    mode = args.batch_mode,
	    max_sequence_length_context = args.max_context_seq_length,
	    max_sequence_length_sentence = args.max_sentence_seq_length,        
	    delimiter = '\t',
	    first_label = True)
	    
  # For the MSR WikiQA dataset
  elif args.task == 'wikiqa':
	  train_data, valid_data, test_data = load_public_dataset(
	    filename = '../datasets/WikiQACorpus/trainfile.txt',
	    validFilename = '../datasets/WikiQACorpus/devfile.txt',
	    testFilename = '../datasets/WikiQACorpus/testfile.txt',
	    clean_sentences = True,
	    vocabulary_min_count = args.vocab_min_count,
	    mode = args.batch_mode,
	    max_sequence_length_context = args.max_context_seq_length,
	    max_sequence_length_sentence = args.max_sentence_seq_length,        
	    delimiter = '\t',
	    first_label = True)
	    
  elif args.task == 'debater':
      if args.motion_id is None:
          train_data, valid_data, test_data = load_debater_dataset(
            filename = args.filename,
            context_free = False,
            clean_sentences = True,
            vocabulary_min_count = args.vocab_min_count,
            mode = args.batch_mode,
            max_sequence_length_context = args.max_context_seq_length,
            max_sequence_length_sentence = args.max_sentence_seq_length,        
            train_valid_test_split_ratio = [0.6,0.1,0.3])
      else:
          train_data, test_data = load_debater_dataset_LOMO_mode(
            mid = args.motion_id,
            filename = args.filename,
            context_free = False,
            clean_sentences = True,
            vocabulary_min_count = args.vocab_min_count,
            mode = args.batch_mode,
            max_sequence_length_context = args.max_context_seq_length,
            max_sequence_length_sentence = args.max_sentence_seq_length)

  #For MSR Paraphrase Corpus dataset -- *UNUSED*
  elif args.task == 'paraphrase_detection':
	  train_data, valid_data, test_data = load_public_dataset_split_train_valid(
        	filename = '../datasets/MSRParaphraseCorpus/trainfile.txt',
        	testFilename = '../datasets/MSRParaphraseCorpus/testfile.txt',
        	clean_sentences = True,
        	vocabulary_min_count = args.vocab_min_count,
        	mode = args.batch_mode,
        	max_sequence_length_context = args.max_context_seq_length,
        	max_sequence_length_sentence = args.max_sentence_seq_length,        
        	train_valid_split_ratio = [0.7,0.3],
        	delimiter = '\t',
        	first_label = True)

  args.num_classes = train_data.num_classes
  args.vocab_size = train_data.vocabulary_size
  args.max_context_seq_length = train_data.max_sequence_length_context
  args.max_sentence_seq_length = train_data.max_sequence_length_sentence

  # ==================================================
  # TRAINING
  # ==================================================
    
  if args.motion_id is None:
    TrainRNNBiSequenceClassification(args,train_data,valid_data)
  else:
    TrainRNNBiSequenceClassification(args,train_data,test_data)

  # tensorboard --logdir /home/vikasraykar/deep/models/exp_1/summaries/
