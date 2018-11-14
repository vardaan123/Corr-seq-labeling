""" padded data reader for sequence classification tasks
"""

import numpy as np
from deep.utils import build_vocabulary, sentence_to_word_ids

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["PaddedDatasetSequenceClassification"]

class PaddedDatasetSequenceClassification(object):
    """ padded data reader for sequence classification tasks

    :implementation notes:

    pads the sequences with 0 till the maximum length so that all sequences in a batch will have the same length
    """
    def __init__(self, sentences, labels, 
        word_to_id = None, 
        max_sequence_length = None):
        """ encodes the [sentences] into ids using the vocabulary dict [word_to_id], optionally trims large sequences

        :params:
            sentences: list of strings
                the list of sentences
            labels: list of int
                the corresponding class labels
            word_to_id : dict (optional)      
                the dict mapping words to their ids  
            max_sequence_length : int (optional) 
                the maximumn length of the sequence allowed
        """
        self.sentences = sentences        

        if word_to_id is None:
            word_to_id, _ = build_vocabulary(sentences)   
        self._word_to_id = word_to_id 
        self._id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
                              
        self._num_examples = len(sentences)
        self._vocabulary_size = len(word_to_id)   

        if labels is None:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        input_sequences = []
        for sentence in sentences: 
            tokens = sentence_to_word_ids(sentence, 
                word_to_id = self._word_to_id,
                max_sequence_length = max_sequence_length) 
            input_sequences.append(tokens)  

        self._seq_lengths = np.array([len(s) for s in input_sequences],dtype=np.int32)    
        if max_sequence_length is None:
            self._max_seq_length = max(self._seq_lengths)
        else:
            self._max_seq_length = max_sequence_length
        self._max_seq_length = int(self._max_seq_length)    

        self._inputs = np.zeros([self._num_examples, self._max_seq_length], dtype=np.int32)
        for idx,s in enumerate(input_sequences):
            self._inputs[idx,:self._seq_lengths[idx]] = s

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
            inputs: np.int32 - [batch_size, seq_length]
            labels: np.int32 - [batch_size]
            seq_lengths: np.int32 - [batch_size]        
        """

        start = self._index_in_epoch
        
        self._index_in_epoch += batch_size
        
        end = self._index_in_epoch

        inputs = self._inputs[start:end]
        if self.mode == 'train':
            labels = self._labels[start:end]
        else:
            labels = None
        seq_lengths = self._seq_lengths[start:end]

        if self._index_in_epoch >= self._num_examples:
            # finished eopch
            self._epochs_completed += 1
            # Shuffle the data
            perm = list(np.arange(self._num_examples))
            np.random.shuffle(perm)
            self._inputs = self._inputs[perm]
            if self.mode == 'train': self._labels = self._labels[perm]
            self._seq_lengths = self._seq_lengths[perm]

            # Start next epoch
            self._index_in_epoch = 0

        #print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

        return inputs, labels, seq_lengths        

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
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def sequence_lengths(self):
        return self._seq_lengths

    @property
    def word_to_id(self):
        return self._word_to_id

    @property
    def id_to_word(self):
        return self._id_to_word

    @property
    def max_sequence_length(self):
        return self._max_seq_length

    @property
    def epochs_completed(self):
        return self._epochs_completed

