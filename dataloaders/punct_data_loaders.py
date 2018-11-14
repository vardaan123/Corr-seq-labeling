#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
"""

import os
import csv
import numpy as np
import sys
import math
from deep.utils import build_vocabulary, clean_str_teddata
from deep.utils import sentence_to_word_ids, word_ids_to_sentence
from deep.dataloaders import FullDatasetSequenceLabelling, FullCorrelatedDatasetSequenceLabelling
from deep.dataloaders import RandomLengthDatasetSequenceLabelling 

__author__  = "Vardaan Pahuja,Anirban Laha"
__email__   = "vapahuja@in.ibm.com,anirlaha@in.ibm.com"

__all__ = ["load_ted_dataset","load_single_ted_dataset","load_multitask_dataset"]

def print_summary(sentences,labels):
    """ summary
    """
    print('%d sentences'%(len(sentences)))
    if isinstance(labels[0], list):
        print('%d number of tasks'%(len(labels)))
    
def load_multitask_input_output(filename, labelFilename, clean_sentences):
    sentence_count = len(open(filename,'r').read().split('\n'))
    labels_count = len(open(labelFilename,'r').read().split('\n'))
    
    print sentence_count,labels_count

    num_tasks = int(math.ceil(labels_count*1.0/sentence_count))

    sentences = []
    labels = []

    for i in range(num_tasks):
        labels.append([])

    f=open(filename,'r')
    for line in f:
        sentence = line.strip()
        if clean_sentences:
            sentence = clean_str_teddata(sentence)
        sentences.append(sentence)
    f.close()

    #label filename is already in the form of tokenids (starting from 1)
    f=open(labelFilename,'r')
    for i,line in enumerate(f):

        label = line.strip()
        labels[i % num_tasks].append(label)
    f.close()
        
    return sentences,labels,num_tasks

  

def load_input_output(filename, labelFilename, clean_sentences):
    sentences = []
    labels = []


    f=open(filename,'r')
    for line in f:
        sentence = line.strip()
        if clean_sentences:
            sentence = clean_str_teddata(sentence)
        sentences.append(sentence)
    f.close()

    #label filename is already in the form of tokenids (starting from 1)
    f=open(labelFilename,'r')
    for line in f:
        label = line.strip()
        labels.append(label)
    f.close()
        
    return sentences,labels

def load_single_ted_dataset(filename,
    labelFilename,
    mode='full',
    max_seq_length=None,
    clean_sentences = False,
    vocabulary_min_count = 5):
    """ data reader for 

    :params:
        mode: can be 'full', 'truncate', 'split', 'random_split'.
    """

    test_sentences, test_labels = load_input_output(filename, labelFilename, clean_sentences)

    print('-----------------------------------------------------dataset summary')
    print('-----------------------------------------------------test')
    print_summary(test_sentences,test_labels)

    # build the vocabulary using the full data      
    print('-----------------------------------------------------building the vocabulary') 
    vocabulary_sentences = list(test_sentences)

    word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count = vocabulary_min_count, tokenizer='simple')

    # encode the dataset
    if mode == 'full':
        test_data  = FullDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id)
    
    elif mode == 'random':
        test_data  = RandomLengthDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id)
            
    elif mode == 'truncate':
        test_data  = FullDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)
    
    return test_data
    
    
def load_multitask_dataset(filename,
    devFilename,
    testFilename,
    labelFilename,
    labelDevFilename,
    labelTestFilename,
    mode='full',
    max_seq_length=None,
    clean_sentences = False,
    vocabulary_min_count = 5):
    """ data reader for correlated tasks

    :params:
        mode: can be 'full','truncate'.
    """


    train_sentences, train_labels, num_tasks = load_multitask_input_output(filename, labelFilename, clean_sentences)
    valid_sentences, valid_labels, valid_num_tasks = load_multitask_input_output(devFilename, labelDevFilename, clean_sentences)
    assert valid_num_tasks == num_tasks
    test_sentences, test_labels, test_num_tasks = load_multitask_input_output(testFilename, labelTestFilename, clean_sentences)
    assert test_num_tasks == num_tasks
    
                        
    print('-----------------------------------------------------dataset summary')
    print('-----------------------------------------------------train')
    print_summary(train_sentences,train_labels)

    print('-----------------------------------------------------valid')
    print_summary(valid_sentences,valid_labels)

    print('-----------------------------------------------------test')
    print_summary(test_sentences,test_labels)

    # build the vocabulary using the full data      
    print('-----------------------------------------------------building the vocabulary') 
    vocabulary_sentences = list(train_sentences)
    vocabulary_sentences.extend(test_sentences)
    vocabulary_sentences.extend(valid_sentences)

    word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count = vocabulary_min_count, tokenizer='simple')

    #encode the dataset
    if mode == 'full':
        train_data = FullCorrelatedDatasetSequenceLabelling(train_sentences,train_labels,num_tasks,
            word_to_id = word_to_id)

        valid_data = FullCorrelatedDatasetSequenceLabelling(valid_sentences,valid_labels,num_tasks,
            word_to_id = word_to_id)

        test_data  = FullCorrelatedDatasetSequenceLabelling(test_sentences,test_labels,num_tasks,
            word_to_id = word_to_id)
            
    elif mode == 'truncate':
        train_data = FullCorrelatedDatasetSequenceLabelling(train_sentences,train_labels,num_tasks,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)

        valid_data = FullCorrelatedDatasetSequenceLabelling(valid_sentences,valid_labels,num_tasks,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)

        test_data  = FullCorrelatedDatasetSequenceLabelling(test_sentences,test_labels,num_tasks,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)
            
    
    return train_data, valid_data, test_data
    

def load_single_multitask_dataset(
    testFilename,
    labelTestFilename,
    mode='full',
    max_seq_length=None,
    clean_sentences = False,
    vocabulary_min_count = 5):
    """ data reader for correlated tasks

    :params:
        mode: can be 'full','truncate'.
    """

    test_sentences, test_labels, num_tasks = load_multitask_input_output(testFilename, labelTestFilename, clean_sentences)
    
                        
    print('-----------------------------------------------------dataset summary')

    print('-----------------------------------------------------test')
    print_summary(test_sentences,test_labels)

    # build the vocabulary using the full data      
    print('-----------------------------------------------------building the vocabulary') 
    vocabulary_sentences = list(test_sentences)

    word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count = vocabulary_min_count, tokenizer='simple')

    #encode the dataset
    if mode == 'full':

        test_data  = FullCorrelatedDatasetSequenceLabelling(test_sentences,test_labels,num_tasks,
            word_to_id = word_to_id)
            
    elif mode == 'truncate':

        test_data  = FullCorrelatedDatasetSequenceLabelling(test_sentences,test_labels,num_tasks,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)
            
    
    return test_data

    
def load_ted_dataset(filename,
    devFilename,
    testFilename,
    labelFilename,
    labelDevFilename,
    labelTestFilename,
    mode='full',
    max_seq_length=None,
    clean_sentences = False,
    vocabulary_min_count = 5):
    """ data reader for 

    :params:
        mode: can be 'full', 'truncate', 'split', 'random_split'.
    """

    train_sentences, train_labels = load_input_output(filename, labelFilename, clean_sentences)
    valid_sentences, valid_labels = load_input_output(devFilename, labelDevFilename, clean_sentences)
    test_sentences, test_labels = load_input_output(testFilename, labelTestFilename, clean_sentences)
    
                        
    print('-----------------------------------------------------dataset summary')
    print('-----------------------------------------------------train')
    print_summary(train_sentences,train_labels)

    print('-----------------------------------------------------valid')
    print_summary(valid_sentences,valid_labels)

    print('-----------------------------------------------------test')
    print_summary(test_sentences,test_labels)

    # build the vocabulary using the full data      
    print('-----------------------------------------------------building the vocabulary') 
    vocabulary_sentences = list(train_sentences)
    vocabulary_sentences.extend(test_sentences)
    vocabulary_sentences.extend(valid_sentences)

    word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count = vocabulary_min_count, tokenizer='simple')

    # encode the dataset
    if mode == 'full':
        train_data = FullDatasetSequenceLabelling(train_sentences,train_labels,
            word_to_id = word_to_id)

        valid_data = FullDatasetSequenceLabelling(valid_sentences,valid_labels,
            word_to_id = word_to_id)

        test_data  = FullDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id)
    
    elif mode == 'random':
        train_data = RandomLengthDatasetSequenceLabelling(train_sentences,train_labels,
            word_to_id = word_to_id)

        valid_data = RandomLengthDatasetSequenceLabelling(valid_sentences,valid_labels,
            word_to_id = word_to_id)

        test_data  = RandomLengthDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id)
            
    elif mode == 'truncate':
        train_data = FullDatasetSequenceLabelling(train_sentences,train_labels,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)

        valid_data = FullDatasetSequenceLabelling(valid_sentences,valid_labels,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)

        test_data  = FullDatasetSequenceLabelling(test_sentences,test_labels,
            word_to_id = word_to_id,
            max_sequence_length = max_seq_length)
            
    
    return train_data, valid_data, test_data
