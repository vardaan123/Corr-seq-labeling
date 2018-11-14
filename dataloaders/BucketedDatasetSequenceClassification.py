""" bucketed data reader for sequence classification tasks
"""

import numpy as np
from deep.utils import build_vocabulary, sentence_to_word_ids

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["BucketedDatasetSequenceClassification"]

class BucketedDatasetSequenceClassification(object):
    """ data reader for sequence classification tasks

    :implementation notes:

    1. buckets the sequences by their length so that all sequences in a batch will have the same length
    2. results in variable size batches
    3. you can specificy the maximum batch size, so that large batches will be split into smaller ones

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
        # build the vocabulary if not provided
        if word_to_id is None:
            word_to_id, id_to_word = build_vocabulary(sentences)   
        self._word_to_id = word_to_id 
        self._id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
                      
        self._num_classes = len(set(labels))
        self._num_examples = len(sentences)
        self._vocabulary_size = len(word_to_id)        
        self._labels = np.array(labels,dtype=np.int32)

        # bucket the sequences
        bucketed_input_sequences = {}
        bucketed_labels = {}
        for i in xrange(self._num_examples): 
            tokens = sentence_to_word_ids(sentences[i], word_to_id = word_to_id)
            n = len(tokens)
            if n not in bucketed_input_sequences:
                bucketed_input_sequences[n] = []
                bucketed_labels[n] = []
            bucketed_input_sequences[n].append(tokens)
            bucketed_labels[n].append(labels[i])

        self._max_seq_length = max(bucketed_input_sequences.keys())    
        if max_sequence_length is not None:
            self._max_seq_length = min(self._max_seq_length,max_sequence_length)
        
        self._bucketed_input_sequences = []
        self._bucketed_labels = []
        for key in bucketed_input_sequences:    
            self._bucketed_input_sequences.append(bucketed_input_sequences[key][:self._max_seq_length])
            self._bucketed_labels.append(bucketed_labels[key])    
        self._num_buckets = len(self._bucketed_input_sequences)            

        self._epochs_completed = 0        
        self._bucket_index_in_epoch = 0
        self._index_in_bucket = 0

        self._bucket_perm = list(np.arange(self._num_buckets))

    def next_batch(self, batch_size):
        """ return the next <=[batch_size] examples from this data set
        all sequences in a batch will have the same sequence length

        :params:
            batch_size: int
                the maximum batch size (note that since this is the bucketed mode the size can be less than the batch_size)

        :returns:
            inputs: np.int32 - [batch_size, seq_length]
            labels: np.int32 - [batch_size]
            seq_lengths: np.int32 - [batch_size]        
        """

        if self._bucket_index_in_epoch >= self._num_buckets:
            # finished epoch, reset everything
            self._epochs_completed += 1
            self._bucket_index_in_epoch = 0
            self._index_in_bucket = 0
            # shuffle the buckets
            np.random.shuffle(self._bucket_perm)
            self._bucketed_input_sequences = [self._bucketed_input_sequences[i] for i in self._bucket_perm]
            self._bucketed_labels = [self._bucketed_labels[i] for i in self._bucket_perm]
            # shuffle the data in each bucket
            for i in xrange(self._num_buckets):
                perm = list(np.arange(len(self._bucketed_input_sequences[i])))
                self._bucketed_input_sequences[i] = [self._bucketed_input_sequences[i][j] for j in perm] 
                self._bucketed_labels[i] = [self._bucketed_labels[i][j] for j in perm] 
        
        current_bucket_size = len(self._bucketed_input_sequences[self._bucket_index_in_epoch])

        if current_bucket_size > batch_size:
            # further split the bucket
            start = self._index_in_bucket
            self._index_in_bucket += batch_size
            end = self._index_in_bucket
            #print('--epoch %d bucket %d (%d) start %d end %d'%(self._epochs_completed,self._bucket_index_in_epoch,current_bucket_size,start,end))
            inputs = np.array(self._bucketed_input_sequences[self._bucket_index_in_epoch][start:end], dtype=np.int32)
            labels = np.array(self._bucketed_labels[self._bucket_index_in_epoch][start:end], dtype=np.int32)

            if (self._index_in_bucket >= current_bucket_size):
                # finshed with this bucket
                self._bucket_index_in_epoch += 1
                self._index_in_bucket = 0
        else:
            # return the current bucket
            #print('--eopch %d bucket %d (%d)'%(self._epochs_completed,self._bucket_index_in_epoch,current_bucket_size))
            inputs = np.array(self._bucketed_input_sequences[self._bucket_index_in_epoch], dtype=np.int32)
            labels = np.array(self._bucketed_labels[self._bucket_index_in_epoch], dtype=np.int32)
            self._bucket_index_in_epoch += 1

        (batch_size,seq_length) = inputs.shape
        seq_lengths = seq_length*np.ones(batch_size,dtype=np.int32)

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
    def bucketed_inputs(self):
        return self._bucketed_input_sequences

    @property
    def bucketed_labels(self):
        return self._bucketed_labels

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

