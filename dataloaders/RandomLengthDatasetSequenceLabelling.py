#!/usr/bin/python
# -*- coding: utf-8 -*-
""" padded data reader for sequence labelling tasks
<TODO> Storing start locations.
"""

import numpy as np
from deep.utils import build_vocabulary, sentence_to_word_ids, simple_tokenize
from nltk.tokenize import word_tokenize
import math, random, sys

__author__  = "Vardaan Pahuja"
__email__   = "vapahuja@in.ibm.com"

__all__ = ["RandomLengthDatasetSequenceLabelling"]

class RandomLengthDatasetSequenceLabelling(object):
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

        """
        *************************************************************************************************************
        """
        #<TO DO> make all change to 'sentences' and 'labels' here, post processing should remain same as FullDatasetSequenceLabelling.py
        new_sentences=[]
        new_labels=[]

        thresh_seq_length = 50 # hyperparameter to be tuned

        for sentence_id,sentence in enumerate(sentences):
            # print 'Sentence_id= %d'%(sentence_id)
            sentence_tokenized = simple_tokenize(sentence)
            if len(sentence_tokenized)<1:
                continue
            sentence_label = labels[sentence_id]
            sentence_label_tokenized = simple_tokenize(sentence_label)     
            
            if len(sentence_tokenized) <= thresh_seq_length:
                new_sentences.append(sentence)
                new_labels.append(sentence_label)
                continue

            try:
                assert len(sentence_tokenized)==len(sentence_label_tokenized)-1
            except AssertionError:
                print 'AssertionError occured'
                print len(sentence_tokenized),len(sentence_label_tokenized)
                print sentence_tokenized
                print sentence_label_tokenized
                continue # discard that particular sentence label pair

            expand_factor = int(math.ceil(len(sentence_tokenized)/20)) # hyperparameter to be tuned

            # print sentence_tokenized
            for i in range(expand_factor):
                
                start_loc = random.randint(0,len(sentence_tokenized)-thresh_seq_length)
                new_seq_len = random.randint(math.floor(thresh_seq_length/2),thresh_seq_length)

                new_sentence_tokenized = sentence_tokenized[start_loc:start_loc + new_seq_len - 1]
                new_sentence_label_tokenized = sentence_label_tokenized[start_loc:start_loc + new_seq_len]
                new_sentence = ' '.join(new_sentence_tokenized)
                new_sentence_label = ' '.join(new_sentence_label_tokenized)
                new_sentences.append(new_sentence)
                new_labels.append(new_sentence_label)
                # print 'length of new sentences= %d'%(len(new_sentences))

        random_idx_perm = list(np.arange(len(new_sentences)))
        np.random.shuffle(random_idx_perm)
        if len(new_sentences)==0:
            print "Null new sentence encountered"

        sentences = [new_sentences[item] for item in random_idx_perm]
        labels = [new_labels[item] for item in random_idx_perm]
        if len(sentences)==0:
            print "Null sentence encountered"

        """
        *************************************************************************************************************
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

        assert len(sentences)>0

        for sentence in sentences: 
            tokens = sentence_to_word_ids(sentence, 
                word_to_id = self._word_to_id,
                max_sequence_length = max_sequence_length, tokenizer='simple') 
            self._input_sequences.append(tokens)  

        assert len(self._input_sequences)>0

        self._seq_lengths = np.array([len(s) for s in self._input_sequences],dtype=np.int32)    
        if max_sequence_length is None:
            self._max_seq_length = max(self._seq_lengths)
        else:
            self._max_seq_length = max_sequence_length
        self._max_seq_length = int(self._max_seq_length)    
        print self._max_seq_length

        # print '**********************'
        # print len(self._input_sequences[0])
        # print self._seq_lengths[0]
        # print len(labels[0])
        # print len(simple_tokenize(labels[0]))
        # sys.stdin.read(1)
        # print '**********************'

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
                print idx
                tokens=simple_tokenize(label)
                #tokens=[int(l)-1 for l in tokens]
                tokens=[int(l) for l in tokens]
                # tokens.append(0) # corresponding label for EOS
                try:
                    self._labels[idx,:self._seq_lengths[idx]] = tokens
                except:
                    print tokens
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
        batch_max_seq_length = np.max(seq_lengths)
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

