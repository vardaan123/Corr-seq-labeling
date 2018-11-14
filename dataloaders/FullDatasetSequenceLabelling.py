#!/usr/bin/python
# -*- coding: utf-8 -*-
""" padded data reader for sequence labelling tasks
"""

import numpy as np
from deep.utils import build_vocabulary, sentence_to_word_ids, simple_tokenize
from nltk.tokenize import word_tokenize

__author__  = "Anirban Laha"
__email__   = "anirlaha@in.ibm.com"

__all__ = ["FullDatasetSequenceLabelling"]

class FullDatasetSequenceLabelling(object):
    """ padded data reader for 
    """
    def __init__(self, sentences, labels, 
        word_to_id = None, 
        max_sequence_length = None):
        """ encodes the [sentences] into ids using the vocabulary dict [word_to_id], optionally trims large sequences

        :params:
            sentences: list of strings
            labels: list of strings (containing sequence of labels)
        """
        self.sentences = sentences        

        if word_to_id is None:
            word_to_id, _ = build_vocabulary(sentences, tokenizer='simple')   
        self._word_to_id = word_to_id 
        self._id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
                              
        self._num_examples = len(sentences)
        self._vocabulary_size = len(word_to_id)   

        if labels is None:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        self._input_sequences = []
        for sentence in sentences: 
            tokens = sentence_to_word_ids(sentence, 
                word_to_id = self._word_to_id,
                max_sequence_length = max_sequence_length, tokenizer='simple') 
            self._input_sequences.append(tokens)  

        self._seq_lengths = np.array([len(s) for s in self._input_sequences],dtype=np.int32)    
        if max_sequence_length is None:
            self._max_seq_length = max(self._seq_lengths)
        else:
            self._max_seq_length = max_sequence_length
        self._max_seq_length = int(self._max_seq_length)    
        print self._max_seq_length

        self._inputs = np.zeros([self._num_examples, self._max_seq_length], dtype=np.int32)
        for idx,s in enumerate(self._input_sequences):
            self._inputs[idx,:self._seq_lengths[idx]] = s
        #self._input_sequences = input_sequences

        if self.mode == 'train':
            #Processing labels now - padding extra ones with 0.
            self._labels = np.zeros([self._num_examples, self._max_seq_length], dtype=np.int32)
            self._num_classes = 0
            print self._labels.shape
            for idx,label in enumerate(labels):
                tokens=simple_tokenize(label)
                #tokens=[int(l)-1 for l in tokens]
                tokens=[int(l) for l in tokens]
                self._labels[idx,:self._seq_lengths[idx]] = tokens[:self._max_seq_length]
                self._num_classes = max(self._num_classes,max(tokens))
            
            self._num_classes += 1 #To account for 0-padded labels as 0-class. Our prediction classes are from [1, max(labels)]
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

        seq_lengths = self._seq_lengths[start:end]
        batch_max_seq_length = np.minimum(np.max(seq_lengths),self._max_seq_length)
        seq_lengths = np.minimum(seq_lengths,np.full(seq_lengths.shape,batch_max_seq_length))
        inputs = self._inputs[start:end,:batch_max_seq_length]
        if self.mode == 'train':
            labels = self._labels[start:end,:batch_max_seq_length]
        else:
            labels = None

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
        print "Num classes:" + str(self._num_classes)
        return self._num_classes

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_sequences(self):
        return self._input_sequences

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

