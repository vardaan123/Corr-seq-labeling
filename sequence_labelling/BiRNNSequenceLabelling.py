""" Bidirectional RNN model for sequence labelling
    TODO(anirlaha): Multilayer+dropout, BiRNN -- full working version.
    TODO(anirlaha): cost-sensitive loss.
"""

import numpy as np
import tensorflow as tf    

__author__  = "Vardaan Pahuja"
__email__   = "vapahuja@in.ibm.com"

__all__ = ["BiRNNSequenceLabelling"]

class BiRNNSequenceLabelling():
  """ RNN model for sequence labelling
  """

  def __init__(self,
    num_classes,
    vocab_size,
    max_seq_length,
    bidirectional,
    embedding_size,
    rnn_size,
    num_layers,
    model,
    init_scale_embedding = 1.0,
    init_scale = 0.1,
    train_embedding_matrix = True,
    dynamic_unrolling = True):
    """ initialize the parameters of the RNN model

    :params:
      num_classes : int
        number of classes 
      vocab_size : int  
        size of the vocabulary     
      max_seq_length : int     
        maximum sequence length allowed
      bidirectional : bool
        use bidirectional RNN if True
      embedding_size : int 
        size of the word embeddings 
      rnn_size : int
        size of RNN hidden state  
      num_layers : int
        number of layers in the RNN 
      model : str
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
      dynamic_unrolling : boolean
        if True dynamic calculation is performed.
        This method of calculation does not compute the RNN steps past the maximum sequence length of the minibatch,
        and properly propagates the state at an example's sequence lengthto the final state output.
        (default: True)
    """

    self.num_classes = num_classes
    self.vocab_size = vocab_size
    self.max_seq_length = max_seq_length
    self.bidirectional = bidirectional
    self.embedding_size = embedding_size
    self.rnn_size = rnn_size
    self.num_layers = num_layers
    self.model = model
    self.init_scale = init_scale
    self.init_scale_embedding = init_scale_embedding
    self.train_embedding_matrix = train_embedding_matrix
    self.dynamic_unrolling = dynamic_unrolling

  def inference(self, seq_inputs, seq_lengths, 
    rnn_dropout_keep_prob, 
    output_dropout_keep_prob):
    """ builds the model 

    :params:
      seq_inputs: tensor, int32 - [batch_size, max_seq_length]
        input sequence encoded as word ids, zero padded       
      seq_lengths: tensor, int32 - [batch_size]
        length of each input sequence
      rnn_dropout_keep_prob: tensor, float32  
        dropout keep probability when using multiple RNNs
      output_dropout_keep_prob: tensor, float32  
        dropout keep probability for the output (i.e. before the softmax layer)

    :returns:    
      logits: tensor, float32 - [batch_size, max_seq_length, num_classes]
        softmax on top of logits will give predictions tensor

    """

    batch_size = tf.size(seq_lengths)

    # choose the RNN cell
    if self.model == 'rnn':
      cell_fw = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
      cell_bw = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
    elif self.model == 'gru':
      cell_fw = tf.nn.rnn_cell.GRUCell(self.rnn_size)
      cell_bw = tf.nn.rnn_cell.GRUCell(self.rnn_size)
    elif self.model == 'basic_lstm':
      cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
      cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
    elif self.model == 'lstm':
      cell_fw = tf.nn.rnn_cell.LSTMCell(self.rnn_size) 
      cell_bw = tf.nn.rnn_cell.LSTMCell(self.rnn_size)      
    else:
      raise Exception("model type not supported: {}".format(self.model))
    
    # dropout
    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=rnn_dropout_keep_prob)
    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=rnn_dropout_keep_prob)

    # multilayer RNN  
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers)

    # initial state
    # initial_state = cell_fw.zero_state(batch_size, tf.float32)

    # word embeddings
    with tf.device("/cpu:0"), tf.name_scope("embedding"):
      # embedding_matrix is exposed outside so that it can be custom intialized
      self.embedding_matrix = tf.Variable(
        tf.random_uniform([self.vocab_size, self.embedding_size],-self.init_scale_embedding, self.init_scale_embedding),
        name="W",
        trainable=self.train_embedding_matrix) 
      
      # Turning this off as moving to dynamic_rnn instead of rnn
      # inputs = tf.split(1, self.max_seq_length, tf.nn.embedding_lookup(self.embedding_matrix, seq_inputs))
      # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
      inputs = tf.nn.embedding_lookup(self.embedding_matrix, seq_inputs) #shape: [batch_size, max_seq_length, embedding_size] for dynamic_rnn
      
    # RNN encoding
    if self.dynamic_unrolling:
      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=inputs,  
        sequence_length=tf.cast(seq_lengths,tf.int64),dtype=tf.float32)    #outputs is 2-tuple, each element of shape: [batch_size, max_seq_length, rnn_size]
      outputs = tf.concat(2,outputs) #resultant outputs shape: [batch_size, max_seq_length, 2, rnn_size]
    # else: #Non-dynamic disabled
      # outputs, states = tf.nn.rnn(cell, inputs, 
        # initial_state=initial_state)    

    # dropout layer
    # with tf.name_scope("dropout"): Turning this off -- moving to dynamic_rnn
      # outputs_drop = [tf.nn.dropout(output, keep_prob=output_dropout_keep_prob) for output in outputs]
    with tf.name_scope("dropout"):
      outputs_drop = tf.nn.dropout(outputs, keep_prob=output_dropout_keep_prob)
      
    # Create predictions for each time step
    with tf.name_scope('softmax'):      
      weights = tf.Variable(tf.truncated_normal([2*self.rnn_size, self.num_classes], mean=0.0, stddev=0.01),
        name='weights')
      biases  = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
        name='biases')
      
      # Moving to dynamic_rnn
      # logits = [tf.nn.xw_plus_b(output_drop, weights, biases) for output_drop in outputs_drop]
      outputs_flat = tf.reshape(outputs_drop, [-1, 2*self.rnn_size])
      logits_flat = tf.nn.xw_plus_b(outputs_flat, weights, biases)
      logits = tf.reshape(logits_flat, [batch_size, -1, self.num_classes])
      
    return logits

  def prediction(self, logits):
      logits_flat = tf.reshape(logits, [-1, self.num_classes])
      predictions_flat = tf.nn.softmax(logits_flat)
      return tf.reshape(predictions_flat, tf.shape(logits))
    
  def loss(self, logits, seq_lengths, labels):
    """ calculates the loss (cross-entropy)

    :params:
      logits: predictions tensor, float - [batch_size, max_seq_length, num_classes]
      seq_lengths: tensor, int32 - [batch_size] - length of each input sequence
      labels: labels tensor, int32 - [batch_size, max_seq_length], with values in the range [0, num_classes],
              0-class implied zero padding.

    :returns:
      loss: loss tensor, float
    """
    
    with tf.name_scope("loss"):
      # batch_size = tf.size(seq_lengths)
      # labels = tf.expand_dims(labels, 2)
      # indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
      # concated = tf.concat(1, [indices, labels])
      # onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.num_classes]), 1.0, 0.0)

      # cross_entropy = -onehot_labels*tf.log(predictions) 
      # loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
      mask = tf.sign(tf.to_float(labels))
      masked_losses = mask * losses
      mean_loss_by_example = tf.div(tf.reduce_sum(masked_losses, reduction_indices=1),tf.to_float(seq_lengths))

    return tf.reduce_mean(mean_loss_by_example)     

  '''
  def loss_pos_weighted(self,logits,labels):
    """ calculates the positive-weighted cross-entropy loss (cost-sensitive)

    :params:
      logits: predictions tensor, float - [batch_size, num_classes]
      labels: labels tensor, int32 - [batch_size], with values in the range [0, num_classes)

    :returns:
      loss: loss tensor, float

    """
    if self.num_classes > 2:
        raise Exception("This is only for binary classification problem")
    with tf.name_scope("loss"):
      pos_weight = 32
      batch_size = tf.size(labels)
      labels = tf.expand_dims(labels, 1)
      indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
      concated = tf.concat(1, [indices, labels])
      onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.num_classes]), 1.0, 0.0)
      
      cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits, onehot_labels, pos_weight)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean_posweighted')

    return loss
'''
  def training(self,loss,
    optimizer = 'adam',
    learning_rate = 1e-3):
    """ sets up the training ops

    :params:
      loss: loss tensor, from loss()
      optimizer: str
        gradient_descent, adam
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
    else:
      raise Exception("optimizer type not supported: {}".format(optimizer))
  
    # use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    #train_op = optimizer.minimize(loss, global_step=global_step)

    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op

  def evaluation(self,logits, labels):
    """ evaluation metric (accuracy)

   :params:
      logits: logits tensor, float - [batch_size, max_seq_length, num_classes]
      labels: labels tensor, int32 - [batch_size, max_seq_length], with values in the range [0, num_classes]

    :returns:
      accuracy: accuracy tensor, float

    """
    with tf.name_scope("accuracy"):
      logits_flat = tf.reshape(logits, [-1, self.num_classes])
      predictions_flat = tf.nn.softmax(logits_flat)
      labels_flat = tf.reshape(labels, [-1])
      mask = tf.sign(tf.to_float(labels_flat))
      num_predictions = tf.reduce_sum(mask)
      error = tf.cast(tf.not_equal(tf.argmax(predictions_flat, 1), tf.to_int64(labels_flat)) , tf.float32)
      error *= mask
      error_rate = tf.reduce_sum(error)/num_predictions

    return 1-error_rate
