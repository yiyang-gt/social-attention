import numpy as np
import scipy.sparse as sp
import cPickle
from collections import defaultdict
import sys, re, os, logging
import pandas as pd

logger = logging.getLogger("social_attention.procdata")

def build_data(fnames):
    """
    Loads and process data.
    """
    revs = []
    vocab = defaultdict(float)
    ins_idx = 0
    user_vocab = set()

    corpuses = []
    for fname in fnames: corpuses += get_corpus(fname)
    for i, corpus in enumerate(corpuses):
        for [tid, uid, label, words] in corpus:
            user_vocab.add(uid)
            for word in words:
                vocab[word] += 1
            datum  = {"y":label,
                      "uid":uid,
                      "words": words,
                      "num_words": len(words),
                      "split": i}
            revs.append(datum)
            ins_idx += 1

    max_l = np.max(pd.DataFrame(revs)["num_words"])

    logger.info("finish building data: %d tweets and %d users" %(ins_idx, len(user_vocab)))
    logger.info("vocab size: %d, max tweet length: %d" %(len(vocab), max_l))
    return revs, vocab, user_vocab, max_l
   
def get_corpus(fname, split=False):
    label2idx = {"positive":1, "negative":0, "neutral":2}
    labeled_corpus = [[],[],[]]
    with open(fname, "rb") as f:
        for line in f:
            parts = line.strip().lower().split()
            tid, uid, label, words = parts[0], parts[1], parts[2], parts[3:]
            label = label2idx[label]
            labeled_corpus[label].append([tid, uid, label, words])
    if split:
        len0, len1, len2 = int(len(labeled_corpus[0])*.8), int(len(labeled_corpus[1])*.8), int(len(labeled_corpus[2])*.8)
        corpus1 = labeled_corpus[0][:len0] + labeled_corpus[1][:len1] + labeled_corpus[2][:len2]
        corpus2 = labeled_corpus[0][len0:] + labeled_corpus[1][len1:] + labeled_corpus[2][len2:]
        return [corpus1, corpus2]
    else:
        corpus = labeled_corpus[0] + labeled_corpus[1] + labeled_corpus[2]
        return [corpus]

class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, random=1):
        self.random = random
        if binary == 1:
            word_vecs, self.k = self.load_bin_vec(fname, vocab)
        else:
            word_vecs, self.k = self.load_txt_vec(fname, vocab)
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))            
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, layer1_size
    
    def load_txt_vec(self, fname, vocab):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, layer1_size

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs: #and vocab[word] >= min_df:
                if self.random:
                    word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
                else:
                    word_vecs[word] = np.zeros(k)
    

if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    w2vfname, ofname, fnames = sys.argv[1], sys.argv[2], sys.argv[3:]

    revs, vocab, user_vocab, max_l = build_data(fnames)

    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(w2vfname, vocab, binary=0)

    cPickle.dump([revs, wordvecs, user_vocab, max_l], open(ofname, "wb"))
    logger.info("dataset created!")
    logger.info("end logging")
    
