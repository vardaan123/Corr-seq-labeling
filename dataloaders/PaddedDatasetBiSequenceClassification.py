""" padded data reader for bi-sequence classification tasks
"""

import numpy as np
from deep.utils import build_vocabulary, sentence_to_word_ids

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["PaddedDatasetBiSequenceClassification"]

class PaddedDatasetBiSequenceClassification(object):
    """ padded data reader for bi-sequence classification tasks

    :implementation notes:

    pads the sequences with 0 till the maximum length so that all sequences in a batch will have the same length

    """
    def __init__(self, contexts, sentences, labels, 
        word_to_id = None, 
        max_sequence_length_context = None,
        max_sequence_length_sentence = None):
        """ encodes the [sentences] into ids using the vocabulary dict [word_to_id], optionally trims large sequences

        :params:
            contexts: list of strings
                the list of context sentences
            sentences: list of strings
                the list of sentences
            labels: list of int
                the corresponding class labels
            word_to_id : dict (optional)      
                the dict mapping words to their ids  
            max_sequence_length_context : int (optional) 
                the maximum length of the sequence allowed for the contexts
            max_sequence_length_sentence : int (optional) 
                the maximum length of the sequence allowed for the sentences               
        """
        self.contexts  = contexts
        self.sentences = sentences        

        if word_to_id is None:
            vocabulary_sentences = list(sentences)
            vocabulary_sentences.extend(contexts)
            word_to_id, _ = build_vocabulary(sentences)   

        self._word_to_id = word_to_id 
        self._id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
                              
        self._num_examples = len(sentences)
        self._vocabulary_size = len(word_to_id)   

        if labels is None:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        # tokenize the sentences
        input_sequences_sentence = []
        for sentence in sentences: 
            tokens = sentence_to_word_ids(sentence, 
                word_to_id = self._word_to_id,
                max_sequence_length = max_sequence_length_sentence) 
            input_sequences_sentence.append(tokens)  

        self._seq_lengths_sentence = np.array([len(s) for s in input_sequences_sentence],dtype=np.int32)    
        if max_sequence_length_sentence is None:
            self._max_seq_length_sentence = max(self._seq_lengths_sentence)
        else:
            self._max_seq_length_sentence = max_sequence_length_sentence

        self._inputs_sentence = np.zeros([self._num_examples, self._max_seq_length_sentence], dtype=np.int32)
        for idx,s in enumerate(input_sequences_sentence):
            self._inputs_sentence[idx,:self._seq_lengths_sentence[idx]] = s

        # tokenize the contexts
        input_sequences_context = []
        for sentence in contexts: 
            tokens = sentence_to_word_ids(sentence, 
                word_to_id = self._word_to_id,
                max_sequence_length = max_sequence_length_context) 
            input_sequences_context.append(tokens)  

        self._seq_lengths_context = np.array([len(s) for s in input_sequences_context],dtype=np.int32)    
        if max_sequence_length_context is None:
            self._max_seq_length_context = max(self._seq_lengths_context)
        else:
            self._max_seq_length_context = max_sequence_length_context

        self._inputs_context = np.zeros([self._num_examples, self._max_seq_length_context], dtype=np.int32)
        for idx,s in enumerate(input_sequences_context):
            self._inputs_context[idx,:self._seq_lengths_context[idx]] = s

        # labels
        if self.mode == 'train':
            self._labels = np.array(labels,dtype=np.int32) 
            self._num_classes = len(set(labels))   

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def reset_batch(self,epochs_completed=0):
        """ reset such that the batch starts from the begining        
        """
        self._index_in_epoch = 0
        self._epochs_completed = epochs_completed

    def next_batch(self, batch_size):
        """ return the next [batch_size] examples from this data set

        :params:
            batch_size: int
                the batch size 

        :returns:
            inputs_context: np.int32 - [batch_size, seq_length]
            inputs_sentence: np.int32 - [batch_size, seq_length]
            labels: np.int32 - [batch_size]
            seq_lengths_context: np.int32 - [batch_size]        
            seq_lengths_sentence: np.int32 - [batch_size]        
        """

        start = self._index_in_epoch
        
        self._index_in_epoch += batch_size
        
        end = self._index_in_epoch

        inputs_context = self._inputs_context[start:end]
        inputs_sentence = self._inputs_sentence[start:end]

        if self.mode == 'train':
            labels = self._labels[start:end]
        else:
            labels = None

        seq_lengths_context = self._seq_lengths_context[start:end]
        seq_lengths_sentence = self._seq_lengths_sentence[start:end]

        if self._index_in_epoch >= self._num_examples:
            # finished eopch
            self._epochs_completed += 1
            # Shuffle the data
            perm = list(np.arange(self._num_examples))
            np.random.shuffle(perm)
            self._inputs_context = self._inputs_context[perm]
            self._inputs_sentence = self._inputs_sentence[perm]
            if self.mode == 'train': self._labels = self._labels[perm]
            self._seq_lengths_context = self._seq_lengths_context[perm]
            self._seq_lengths_sentence = self._seq_lengths_sentence[perm]

            # Start next epoch
            self._index_in_epoch = 0

        #print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

        return inputs_context, inputs_sentence, labels, seq_lengths_context, seq_lengths_sentence        

    def next_batch_with_sampling(self, batch_size, sampling_rate=1.0):
        """ return the next [batch_size] examples from this data set

        :params:
            batch_size: int
                the batch size 

        :returns:
            inputs_context: np.int32 - [batch_size, seq_length]
            inputs_sentence: np.int32 - [batch_size, seq_length]
            labels: np.int32 - [batch_size]
            seq_lengths_context: np.int32 - [batch_size]        
            seq_lengths_sentence: np.int32 - [batch_size]        
        """

        
        self._index_in_epoch += batch_size
        while np.random.random_sample() > sampling_rate and self._index_in_epoch < self._num_examples:
            self._index_in_epoch += batch_size    
        
        end = self._index_in_epoch
        start = self._index_in_epoch-batch_size

        inputs_context = self._inputs_context[start:end]
        inputs_sentence = self._inputs_sentence[start:end]

        if self.mode == 'train':
            labels = self._labels[start:end]
        else:
            labels = None

        seq_lengths_context = self._seq_lengths_context[start:end]
        seq_lengths_sentence = self._seq_lengths_sentence[start:end]

        if self._index_in_epoch >= self._num_examples:
            # finished eopch
            self._epochs_completed += 1
            # Shuffle the data
            perm = list(np.arange(self._num_examples))
            np.random.shuffle(perm)
            self._inputs_context = self._inputs_context[perm]
            self._inputs_sentence = self._inputs_sentence[perm]
            if self.mode == 'train': self._labels = self._labels[perm]
            self._seq_lengths_context = self._seq_lengths_context[perm]
            self._seq_lengths_sentence = self._seq_lengths_sentence[perm]

            # Start next epoch
            self._index_in_epoch = 0

        #print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

        return inputs_context, inputs_sentence, labels, seq_lengths_context, seq_lengths_sentence        

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def inputs_sentence(self):
        return self._inputs_sentence

    @property
    def inputs_context(self):
        return self._inputs_context

    @property
    def labels(self):
        return self._labels

    @property
    def seq_lengths_sentence(self):
        return self._seq_lengths_sentence

    @property
    def seq_lengths_context(self):
        return self._seq_lengths_context

    @property
    def word_to_id(self):
        return self._word_to_id

    @property
    def id_to_word(self):
        return self._id_to_word

    @property
    def max_sequence_length_sentence(self):
        return self._max_seq_length_sentence

    @property
    def max_sequence_length_context(self):
        return self._max_seq_length_context

    @property
    def epochs_completed(self):
        return self._epochs_completed

