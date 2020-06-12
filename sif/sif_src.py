import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# The allSIFSimilarities function uses some code from the SIF repo
# All other functions are taken from the SIF repo, with slight modifications and docstrings added

  
def allSIFSimilarities(sentences, word_dict, word_embeddings, 
                       word_weights, rmpc=1):
    """ 
    Compute the SIF-weighted cosine similarity between all sentences in a list
    Similarity scores are between -1 and 1, a score of -1 meaning the sentences are
    completely dissimilar, and 1 meaning the sentences are the same

    """
    
    x, m = sentences2idx(sentences, word_dict) # get word index array and binary mask
    w = seq2weight(x, m, word_weights) # get word weights
    embedding = SIF_embedding(word_embeddings, x, w, rmpc)
    
    sims_matrix = cosine_similarity(embedding)    
   
    return sims_matrix





def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, rmpc):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if  rmpc > 0:
        emb = remove_pc(emb, rmpc)
    return emb



def getWordmap(textfile):
    """
    Convert the word vector file (textfile) into a word dictionary
    """
    words={}
    We = []
    with open(textfile, 'r', encoding='utf-8') as f:
        n = 0
        for line in f:
            # spaces separate the values
            i = line.split(" ")
            # convert the values from str to float
            # i[0] is the word itself
            v = [float(j) for j in i[1:]]
            words[i[0]] = n # word dictionary, key=index
            We.append(v)
            n+=1
    return words, np.array(We)


def prepare_data(list_of_seqs):
    """ 
    list_of_seqs = list of sentences that have been converted to sequences of word indices
    """
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def lookupIDX(words, w):
    """ get word w's index (integer) from the word dictionary """
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1


def seq2weight(seq, mask, weight4ind):
    """ gets the matrix of word weights for all sentences """
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


def getWordWeight(weightfile, a=1e-3):
    if a <= 0: a = 1.0 # a must be > 0
    word2weight = {}
    lines = open(weightfile).readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)
    return word2weight


def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    sentences: a list of sentences
    words: a dictionary, where words['str'] is the indices of the word 'str'
    return: x1, m1. 
      x1[i, :] is the word indices in sentence i,
      m1[i,:] is the mask for sentence i (0 means no word at the location)
    """

    seqs = [[lookupIDX(words, i) for i in s.split()] for s in sentences]
    x1, m1 = prepare_data(seqs)
    return x1, m1
