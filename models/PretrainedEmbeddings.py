""" load the pre-trained [word2vec or glove] embeddings for a given vocabulary

--word2vec

https://code.google.com/archive/p/word2vec/
pre-trained vectors trained on part of Google News dataset (about 100 billion words). 
The model contains 300-dimensional vectors for 3 million words and phrases.

--glove
http://nlp.stanford.edu/projects/glove/
pre-trained vectors trained on Common Crawl dataset (about 840 billion words). 
The model contains 300-dimensional vectors for 2.2M vocab size.

"""

import gensim
import numpy as np
import time
import os

__author__  = "Vikas Raykar,Anirban Laha"
__email__   = "viraykar@in.ibm.com"

__all__ = ["PretrainedEmbeddings"]

class PretrainedEmbeddings():
    """ load the pre-trained embeddings
    """

    def __init__(self,filename):        
        """ load the pre-trained embeddings

        :params:
            filename : string
                full path to the .bin file containing the pre-trained word vectors
                GoogleNews-vectors-negative300.bin
                can be downloaded from https://code.google.com/archive/p/word2vec/
        """

        _, file_extension = os.path.splitext(filename)
        if file_extension == '.bin':
            is_binary = True
        else:
            is_binary = False

        print('Loading the pre-trained embeddings from [%s].'%(filename))
        start = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename,binary=is_binary)
        time_taken = time.time()-start
        print('Takes %2.2f secs !'%(time_taken))

        self._embedding_size = 300

    def load_embedding_matrix(self,word_to_id):
        """ extract the embedding matrix for a given vocabulary

        words not in word2vec are intialized randonaly to uniform(-a,a), where a is chosen such that the
        unknown words have the same variance as words already in word_to_id

        :params:  
            word_to_id: dict
                dict mapping a word to its id, e.g., word_to_id['the'] = 3

        :returns:
            embedding_matrix: np.array, float32 - [vocab_size, embedding_size]
                the embedding matrix         
        """
        vocab_size = len(word_to_id)

        embedding_matrix = 100.0*np.ones((vocab_size,self._embedding_size)).astype('float32')

        count = 0
        for word in word_to_id:
            if word in self.model:
                embedding_matrix[word_to_id[word],:] = self.model[word]
                count += 1
        print('Found pre-trained word2vec embeddings for %d/%d words.'%(count,vocab_size))

        init_OOV = np.sqrt(3.0*np.var(embedding_matrix[embedding_matrix!=100.0]))

        for word in word_to_id:
            if word not in self.model:
                embedding_matrix[word_to_id[word],:] = np.random.uniform(-init_OOV,init_OOV,(1,self._embedding_size))

        return embedding_matrix
    
    @property
    def embedding_size(self):
        return self._embedding_size

if __name__ == '__main__':
    """ example usage
    """

    import os
    from deep.props import *
    
    sentences = []
    sentences.append('He argued that violent video games should be banned .')
    sentences.append('He was a good crow .')
    sentences.append('The individual freedom of expression is therefore essential to the well-being of society.')

    from deep.utils import build_vocabulary
    word_to_id, id_to_word = build_vocabulary(sentences, min_count = 1)

    filename = os.path.join(props['resources_folder'],'GoogleNews-vectors-negative300.bin')
    #filename = os.path.join(props['resources_folder'],'glove.840B.300d.processed.txt')

    embeddings = PretrainedEmbeddings(filename)
    embedding_matrix = embeddings.load_embedding_matrix(word_to_id)




