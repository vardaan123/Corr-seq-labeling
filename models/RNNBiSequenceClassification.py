""" RNN context_model for bisequence classification
    Two sequences in debater means one of them is the context/motion
    This is for context dependent sequence classification
"""

"""
TODO: Only GRUCell and BasicLSTMCell has been tested and tried out. Need to work on others.
"""

import numpy as np
import tensorflow as tf    

__author__  = "Anirban Laha"
__email__   = "anirlaha@in.ibm.com"

__all__ = ["RNNBiSequenceClassification"]

class RNNBiSequenceClassification():
  """ RNN context_model for bisequence classification
  """

  def __init__(self,
    num_classes,
    vocab_size,
    max_context_seq_length,
    max_sentence_seq_length,
    embedding_size,
    context_rnn_size,
    sentence_rnn_size,
    context_num_layers,
    sentence_num_layers,
    context_model,
    sentence_model,
    init_scale_embedding = 1.0,
    init_scale = 0.1,
    train_embedding_matrix = True,
    architecture = "concat",
    dynamic_unrolling = True,
    class_ratios=[0.97,0.03]):
    """ initialize the parameters of the RNN context_model

    :params:
      num_classes : int
        number of classes 
      vocab_size : int  
        size of the vocabulary     
      max_context_seq_length : int     
        maximum sequence length allowed for context
      max_sentence_seq_length : int     
        maximum sequence length allowed for sentence
      embedding_size : int 
        size of the word embeddings 
      context_rnn_size : int
        size of context RNN hidden state  
      sentence_rnn_size : int
        size of sentence RNN hidden state  
      context_num_layers : int
        number of layers in the context RNN 
      sentence_num_layers : int
        number of layers in the sentence RNN 
      context_model : str
        rnn, gru, basic_lstm, or lstm 
      sentence_model : str
        rnn, gru, basic_lstm, or lstm 
      init_scale_embedding : float
        random uniform initialization in the range [-init_scale,init_scale] for the embedding layer
        (default: 1.0)                            
      init_scale : float
        random uniform initialization in the range [-init_scale,init_scale]  
        (default: 0.1)
      train_embedding_matrix : boolean
        if False does not train the embedding matrix and keeps it fixed 
        (default: True)    
      architecture : str
        Variants to try out
            1."conditional" : Two RNNs --> Output of first one (context) is initial state of second one.
            2."conditionedInput" : Two RNNs --> Output of first one (context) is fed as input for second one concatenated with every input.
            3."concat" : Two RNNs --> Output of two RNNs concatenated and predicted using MLP.
            4."quadraticForm" : Two RNNs --> Output of two RNNs --> x and y --> Score predicted by xT W y.
            5."fullConditional" : Two RNNs --> Output of first one (context) is initial state of second one AND Output of first one (context) is fed as input for second one concatenated with every input.
      dynamic_unrolling : boolean
        if True dynamic calculation is performed.
        This method of calculation does not compute the RNN steps past the maximum sequence length of the minibatch,
        and properly propagates the state at an example's sequence lengthto the final state output.
        (default: True)
      class_ratios : list of float, length being same as num_classes
    """

    self.num_classes = num_classes
    self.class_ratios = class_ratios
    self.vocab_size = vocab_size
    self.max_context_seq_length = max_context_seq_length
    self.max_sentence_seq_length = max_sentence_seq_length
    self.embedding_size = embedding_size
    self.context_rnn_size = context_rnn_size
    self.sentence_rnn_size = sentence_rnn_size
    self.context_num_layers = context_num_layers
    self.sentence_num_layers = sentence_num_layers
    self.context_model = context_model
    self.sentence_model = sentence_model
    self.init_scale = init_scale
    self.init_scale_embedding = init_scale_embedding
    self.train_embedding_matrix = train_embedding_matrix
    self.architecture = architecture
    self.dynamic_unrolling = dynamic_unrolling

  def inference(self, context_inputs,
                      context_lengths,
                      sentence_inputs,
                      sentence_lengths, 
                      context_rnn_dropout_keep_prob,
                      sentence_rnn_dropout_keep_prob,
                      output_dropout_keep_prob,
                      return_logit=False):
    """ builds the bisequence architecture 

    :params:
      context_inputs: tensor, int32 - [batch_size, max_context_seq_length]
        context sequence encoded as word ids, zero padded       
      context_lengths: tensor, int32 - [batch_size]
        length of each context sequence
      sentence_inputs: tensor, int32 - [batch_size, max_sentence_seq_length]
        sentence sequence encoded as word ids, zero padded       
      sentence_lengths: tensor, int32 - [batch_size]
        length of each sentence sequence
      context_rnn_dropout_keep_prob: tensor, float32  
        dropout keep probability when using multiple RNNs for context
      sentence_rnn_dropout_keep_prob: tensor, float32  
        dropout keep probability when using multiple RNNs for sentence
      output_dropout_keep_prob: tensor, float32  
        dropout keep probability for the output (i.e. before the softmax layer)

    :returns:    
      predictions: tensor, float32 - [batch_size, num_classes]
        predictions tensor

    """
    #This check should be done in train script which calls inference method
    # if tf.size(context_lengths) != tf.size(sentence_lengths):
        # raise Exception("context_lengths and sentence_lengths should match in size")
    batch_size = tf.size(context_lengths)

    # initializer
    initializer = tf.random_uniform_initializer(-self.init_scale,self.init_scale)

    with tf.variable_scope('context'):
        # choose the context RNN cell
        if self.context_model == 'rnn':
          context_cell = tf.nn.rnn_cell.BasicRNNCell(self.context_rnn_size)
        elif self.context_model == 'gru':
          context_cell = tf.nn.rnn_cell.GRUCell(self.context_rnn_size)
        elif self.context_model == 'basic_lstm':
          context_cell = tf.nn.rnn_cell.BasicLSTMCell(self.context_rnn_size)
        elif self.context_model == 'lstm':
          context_cell = tf.nn.rnn_cell.LSTMCell(self.context_rnn_size)      
        else:
          raise Exception("context_model type not supported: {}".format(self.context_model))

        # dropout
        context_cell = tf.nn.rnn_cell.DropoutWrapper(context_cell, output_keep_prob=context_rnn_dropout_keep_prob)
    
        # multilayer RNN  
        context_cell = tf.nn.rnn_cell.MultiRNNCell([context_cell] * self.context_num_layers)

    with tf.variable_scope('sentence'):
        # choose the sentence RNN cell
        if self.sentence_model == 'rnn':
          sentence_cell = tf.nn.rnn_cell.BasicRNNCell(self.sentence_rnn_size)
        elif self.sentence_model == 'gru':
          sentence_cell = tf.nn.rnn_cell.GRUCell(self.sentence_rnn_size)
        elif self.sentence_model == 'basic_lstm':
          sentence_cell = tf.nn.rnn_cell.BasicLSTMCell(self.sentence_rnn_size)
        elif self.sentence_model == 'lstm':
          sentence_cell = tf.nn.rnn_cell.LSTMCell(self.sentence_rnn_size)      
        else:
          raise Exception("sentence_model type not supported: {}".format(self.sentence_model))
    
        # dropout
        sentence_cell = tf.nn.rnn_cell.DropoutWrapper(sentence_cell, output_keep_prob=sentence_rnn_dropout_keep_prob)

        # multilayer RNN  
        sentence_cell = tf.nn.rnn_cell.MultiRNNCell([sentence_cell] * self.sentence_num_layers)

    # initial state
    context_initial_state = context_cell.zero_state(batch_size, tf.float32)
    if self.architecture != "conditional" and self.architecture != "fullConditional":
        sentence_initial_state = sentence_cell.zero_state(batch_size, tf.float32)

    # word embeddings
    with tf.device("/cpu:0"), tf.name_scope("embedding"):
      # embedding_matrix is exposed outside so that it can be custom intialized
      self.embedding_matrix = tf.Variable(
        tf.random_uniform([self.vocab_size, self.embedding_size],-self.init_scale_embedding, self.init_scale_embedding),
        name="W",
        trainable=self.train_embedding_matrix) 
      
      context_inputs = tf.split(1, self.max_context_seq_length, tf.nn.embedding_lookup(self.embedding_matrix, context_inputs))
      context_inputs = [tf.squeeze(input_, [1]) for input_ in context_inputs]
      sentence_inputs = tf.split(1, self.max_sentence_seq_length, tf.nn.embedding_lookup(self.embedding_matrix, sentence_inputs))
      sentence_inputs = [tf.squeeze(input_, [1]) for input_ in sentence_inputs]
      
    # RNN encoding
    with tf.variable_scope('context'):
    
        if self.dynamic_unrolling:
            context_outputs, context_states = tf.nn.rnn(context_cell, context_inputs, 
                initial_state=context_initial_state, 
                sequence_length=context_lengths) 
        else:
            context_outputs, context_states = tf.nn.rnn(context_cell, context_inputs, 
                initial_state=context_initial_state)  
                
        if (self.context_model == 'lstm') or (self.context_model == 'basic_lstm'):
          # for lstm the state size is twice the rnn size
          context_output = tf.slice(context_states, [0,self.context_rnn_size], [-1, -1])
        else:
          context_output = context_states

    with tf.variable_scope('sentence'):
        if self.architecture != "conditional" and self.architecture != "conditionedInput" and self.architecture != "fullConditional":
            if self.dynamic_unrolling:
                sentence_outputs, sentence_states = tf.nn.rnn(sentence_cell, sentence_inputs, 
                    initial_state=sentence_initial_state, 
                    sequence_length=sentence_lengths) 
            else:
                sentence_outputs, sentence_states = tf.nn.rnn(sentence_cell, sentence_inputs, 
                    initial_state=sentence_initial_state)    

            if (self.sentence_model == 'lstm') or (self.sentence_model == 'basic_lstm'):
              # for lstm the state size is twice the rnn size
              sentence_output = tf.slice(sentence_states, [0,self.sentence_rnn_size], [-1, -1])
            else:
              sentence_output = sentence_states
      
    #For architecture variant "concat"
    if self.architecture == "concat":
        print 'concat'
        output = tf.concat(1, [context_output, sentence_output])
        # dropout layer
        with tf.name_scope("dropout"):
            output_drop = tf.nn.dropout(output, keep_prob=output_dropout_keep_prob)
      
        # linear layer 
        with tf.name_scope('softmax'):      
            weights = tf.Variable(tf.truncated_normal([self.context_rnn_size+self.sentence_rnn_size, self.num_classes], mean=0.0, stddev=0.01),
              name='weights')
            biases  = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
              name='biases')
      
            logits = tf.nn.xw_plus_b(output_drop,weights,biases, name='logits') 
            predictions = tf.nn.softmax(logits, name='predictions')
      
        return predictions
        
    #For architecture variant "quadraticForm"
    elif self.architecture == "quadraticForm":
        print 'quadraticForm'
        # linear layer 
        with tf.name_scope('softmax'):     
            weighttensor = tf.Variable(tf.truncated_normal([self.context_rnn_size, self.sentence_rnn_size, self.num_classes], mean=0.0, stddev=0.01),
              name='weights')
            weights = tf.split(2, self.num_classes, weighttensor)
            weights = [tf.squeeze(wt, [2]) for wt in weights]
            biases  = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
              name='biases')
      
            left = [tf.expand_dims(tf.matmul(context_output, wt), 1) for wt in weights]
            right = tf.expand_dims(sentence_output, 2)
            logits = [tf.squeeze(tf.batch_matmul(l, right), [1,2]) for l in left]
            logits = [logits[i]+biases[i] for i in range(self.num_classes)]
            logits = tf.transpose(tf.pack(logits))
            predictions = tf.nn.softmax(logits, name='predictions')
            
        return predictions
    
    #For architecture variant "conditional"
    elif self.architecture == "conditional":
        print 'conditional'
        sentence_initial_state = context_states

    #For architecture variant "conditionedInput"
    elif self.architecture == "conditionedInput":
        print 'conditionedInput'
        sentence_inputs = [ tf.concat(1, [context_output, input_]) for input_ in sentence_inputs]
        
    #For architecture variant "fullConditional"
    elif self.architecture == "fullConditional":
        print 'fullConditional'
        sentence_initial_state = context_states
        sentence_inputs = [ tf.concat(1, [context_output, input_]) for input_ in sentence_inputs]
    

    #For architecture variant "conditionedInput" and "conditional" and "fullConditional" run rest of the pipeline
    with tf.variable_scope('sentence'):
        if self.dynamic_unrolling:
            sentence_outputs, sentence_states = tf.nn.rnn(sentence_cell, sentence_inputs, 
                initial_state=sentence_initial_state, 
                sequence_length=sentence_lengths)    
        else:
            sentence_outputs, sentence_states = tf.nn.rnn(sentence_cell, sentence_inputs, 
                initial_state=sentence_initial_state)    

        if (self.sentence_model == 'lstm') or (self.sentence_model == 'basic_lstm'):
            # for lstm the state size is twice the rnn size
            sentence_output = tf.slice(sentence_states, [0,self.sentence_rnn_size], [-1, -1])
        else:
            sentence_output = sentence_states

    # dropout layer
    with tf.name_scope("dropout"):
      output_drop = tf.nn.dropout(sentence_output, keep_prob=output_dropout_keep_prob)

    # linear layer 
    with tf.name_scope('softmax'):      
      weights = tf.Variable(tf.truncated_normal([self.sentence_rnn_size, self.num_classes], mean=0.0, stddev=0.01),
        name='weights')
      biases  = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
        name='biases')

      logits = tf.nn.xw_plus_b(output_drop,weights,biases, name='logits') 
      predictions = tf.nn.softmax(logits, name='predictions')

    if return_logit:
        return logits
    print 'Returning'
    return sentence_output,predictions

        
  def loss(self,predictions,labels):
    """ calculates the loss (cross-entropy)

    :params:
      predictions: predictions tensor, float - [batch_size, num_classes]
      labels: labels tensor, int32 - [batch_size], with values in the range [0, num_classes)

    :returns:
      loss: loss tensor, float

    """
    with tf.name_scope("loss"):
      batch_size = tf.size(labels)
      labels = tf.expand_dims(labels, 1)
      indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
      concated = tf.concat(1, [indices, labels])
      onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.num_classes]), 1.0, 0.0)

      cross_entropy = -onehot_labels*tf.log(predictions) 
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss     

  def loss_pos_weighted(self,logits,labels):
    """ calculates the positive-weighted cross-entropy loss (cost-sensitive)

    :params:
      logits: logits tensor, float - [batch_size, num_classes]
      labels: labels tensor, int32 - [batch_size], with values in the range [0, num_classes)

    :returns:
      loss: loss tensor, float

    """
    if self.num_classes > 2:
        raise Exception("This is only for binary classification problem")
    with tf.name_scope("loss"):
      pos_weight = self.class_ratios[0]/self.class_ratios[1]
      batch_size = tf.size(labels)
      labels = tf.expand_dims(labels, 1)
      indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
      concated = tf.concat(1, [indices, labels])
      onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.num_classes]), 1.0, 0.0)
      
      cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits, onehot_labels, pos_weight)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean_posweighted')

    return loss

  def training(self,loss,
    optimizer = 'adam',
    learning_rate = 1e-3):
    """ sets up the training ops

    :params:
      loss: loss tensor, from loss()
      optimizer: str
        gradient_descent, adam, adadelta, rmsprop, adagrad, momentum
      learning_rate : float
        the learning rate  

    :returns:
      train_op: the op for training

    """
    # create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # create the gradient descent optimizer with the given learning rate.
    if optimizer == 'gradient_descent':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5)
    else:
      raise Exception("optimizer type not supported: {}".format(optimizer))
  
    # use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    #train_op = optimizer.minimize(loss, global_step=global_step)

    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op

  def evaluation(self,predictions, labels):
    """ evaluation metric (accuracy)

   :params:
      predictions: predictions tensor, float - [batch_size, num_classes]
      labels: labels tensor, int32 - [batch_size], with values in the range [0, num_classes)

    :returns:
      accuracy: accuracy tensor, float

    """
    with tf.name_scope("accuracy"):
      correct_predictions = tf.nn.in_top_k(predictions, labels, 1)
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    return accuracy  



