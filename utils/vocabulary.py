#!/usr/bin/python
# -*- coding: utf-8 -*-
""" tools to build vocabulary
"""

from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
from clean_str import *

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["build_vocabulary","sentence_to_word_ids","word_ids_to_sentence","simple_tokenize"]

def sentence_to_word_ids(sentence, word_to_id, max_sequence_length = None, tokenizer='nltk'):
    """ encode a given [sentence] to a list of word ids using the vocabulary dict [word_to_id]
    adds a end-of-sentence marker (<EOS>)
    out-of-vocabulary words are mapped to 2    
    """
    if tokenizer=='nltk':
        tokens = word_tokenize(sentence)
    else:
        tokens = simple_tokenize(sentence)

    if max_sequence_length is not None:
        tokens = tokens[:max_sequence_length-1]

    tokens.append('<EOS>')

    return [word_to_id.get(word,2) for word in tokens]
    
def simple_tokenize(string):
    return string.strip().split(' ')

def word_ids_to_sentence(word_ids_list, id_to_word):
    """ decode a given list of word ids [word_ids_list] to a sentence using the inverse vocabulary dict [id_to_word]
    """
    tokens = [id_to_word.get(id) for id in word_ids_list if id >= 2]

    return ' '.join(tokens).capitalize()+'.'

def build_vocabulary(sentences, min_count = 1, tokenizer='nltk'):
    """ build the vocabulary from a list of `sentences'
    uses word_tokenize from nltk for word tokenization

    :params:
        sentences: list of strings
            the list of sentences
        min_count: int
            keep words whose count is >= min_count                 

    :returns:
       word_to_id: dict
            dict mapping a word to its id, e.g., word_to_id['the'] = 3
            the id start from 3
            2 is reserved for out-of-vocabulary words (<OOV>)
            1 is reserved for end-of-sentence marker (<EOS>)
            0 is reserved for padding (<PAD>)
    """
    
    wordcount = Counter()
    for sentence in sentences:
        if tokenizer=='nltk':
            tokens = word_tokenize(sentence)
        else:
            tokens = simple_tokenize(sentence)
        wordcount.update(tokens)

    print('vocabulary size = %d'%(len(wordcount))) 

    # filtering
    count_pairs = wordcount.most_common()
    count_pairs = [c for c in count_pairs if c[1] >= min_count]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(3,len(words)+3)))
    print('vocabulary size = %d (after filtering with min_count =  %d)'%(len(word_to_id),min_count)) 

    word_to_id['<PAD>'] = 0
    word_to_id['<EOS>'] = 1
    word_to_id['<OOV>'] = 2

    id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))

    return word_to_id, id_to_word
    
if __name__ == '__main__':
    string = "it 's a big community it 's such a big community in fact that our annual gathering is the largest scientific meeting in the world over 15,000 scientists go to san francisco every year for that and every one of those scientists is in a research group and every research group studies a wide variety of topics for us at cambridge it 's as varied as the el ni ñ o oscillation which affects weather and climate to the assimilation of satellite data to emissions from crops that produce biofuels which is what i happen to study and in each one of these research areas of which there are even more there are phd students like me and we study incredibly narrow topics things as narrow as a few processes or a few molecules and one of the molecules i study is called isoprene which is here it 's a small organic molecule </s>"
    print string
    print clean_str_teddata(string)
    print word_tokenize(clean_str_teddata(string))
    print simple_tokenize(clean_str_teddata(string))
    

