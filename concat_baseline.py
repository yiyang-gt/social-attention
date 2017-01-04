import sys, cPickle, logging
import numpy as np

from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *
from keras.utils import np_utils

from keras_classes import *
from process_data import WordVecs

logger = logging.getLogger("social_attention.concat.baseline")

def train_conv_net(datasets,                # word indices of train/dev/test tweets
                   U,                       # pre-trained word embeddings
                   A,                       # pre-trained author embeddings
                   text_dim=100,            # dim of sentence vector
                   batch_size=20,           # mini batch size
                   n_epochs=15):
    """
    train and evaluate convolutional neural network model for sentiment clasisification with SemEval datasets
    """
    # prepare datasets
    train_set, dev_set, test13_set, test14_set, test15_set = datasets
    train_set_x, dev_set_x, test13_set_x, test14_set_x, test15_set_x = train_set[:,:-2], dev_set[:,:-2], test13_set[:,:-2], test14_set[:,:-2], test15_set[:,:-2]
    train_set_u, dev_set_u, test13_set_u, test14_set_u, test15_set_u = train_set[:,-2], dev_set[:,-2], test13_set[:,-2], test14_set[:,-2], test15_set[:,-2]
    train_set_y, dev_set_y, test13_set_y, test14_set_y, test15_set_y = train_set[:,-1], dev_set[:,-1], test13_set[:,-1], test14_set[:,-1], test15_set[:,-1]
    train_set_y = np_utils.to_categorical(train_set_y, 3) # negative, positive, neutral

    # build model with keras
    n_tok = len(train_set[0])-2  # num of tokens in a tweet
    vocab_size, emb_dim = U.shape
    user_vocab_size, user_emb_dim = A.shape
    sequence = Input(shape=(n_tok,), dtype='int32')
    author = Input(shape=(1,), dtype='int32')
    emb_layer = Embedding(vocab_size, emb_dim, weights=[U], trainable=False, input_length=n_tok)(sequence) 
    user_emb_layer = Embedding(user_vocab_size, user_emb_dim, weights=[A], trainable=False, input_length=1)(author)
    user_layer = Summation()(user_emb_layer) # 3D to 2D
    conv_layer = Convolution1D(text_dim, 2, activation='tanh')(emb_layer)
    pool_layer = MaxPooling1D(n_tok - 1)(conv_layer)
    text_layer = Flatten()(pool_layer)
    merged_layer = merge([text_layer, user_layer], mode='concat', concat_axis=-1)
    pred_layer = Dense(3, activation='softmax')(merged_layer)
    model = Model(input=[sequence, author], output=pred_layer)

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])

    # start training
    bestDev_perf, best13_perf, best14_perf, best15_perf = 0., 0., 0., 0.
    corr13_perf, corr14_perf, corr15_perf = 0., 0., 0.
    for epo in xrange(n_epochs):
        # training
        model.fit([train_set_x, train_set_u], train_set_y, batch_size=batch_size, nb_epoch=1, verbose=0)

        # evaluation
        ypred = model.predict([dev_set_x, dev_set_u], batch_size=batch_size, verbose=0).argmax(axis=-1)
        dev_perf = avg_fscore(ypred, dev_set_y)

        ypred = model.predict([test13_set_x, test13_set_u], batch_size=batch_size, verbose=0).argmax(axis=-1)
        test13_perf = avg_fscore(ypred, test13_set_y)
        best13_perf = max(best13_perf, test13_perf)
        ypred = model.predict([test14_set_x, test14_set_u], batch_size=batch_size, verbose=0).argmax(axis=-1)
        test14_perf = avg_fscore(ypred, test14_set_y)
        best14_perf = max(best14_perf, test14_perf)
        ypred = model.predict([test15_set_x, test15_set_u], batch_size=batch_size, verbose=0).argmax(axis=-1)
        test15_perf = avg_fscore(ypred, test15_set_y)
        best15_perf = max(best15_perf, test15_perf)

        if dev_perf >= bestDev_perf:
            bestDev_perf, corr13_perf, corr14_perf, corr15_perf = dev_perf, test13_perf, test14_perf, test15_perf

        logger.info("Epoch: %d Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f" %(epo+1, dev_perf*100, test13_perf*100, test14_perf*100, test15_perf*100))

    print("CORR: Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f AVG perf: %.3f" %(bestDev_perf*100, corr13_perf*100, corr14_perf*100, corr15_perf*100, (corr13_perf+corr14_perf+corr15_perf)/3*100))
    print("BEST: Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f" %(bestDev_perf*100, best13_perf*100, best14_perf*100, best15_perf*100))


def avg_fscore(y_pred, y_gold):
    pos_p, pos_g = 0, 0
    neg_p, neg_g = 0, 0
    for p in y_pred:
        if p == 1: pos_p += 1
        elif p == 0: neg_p += 1
    for g in y_gold:
        if g == 1: pos_g += 1
        elif g == 0: neg_g += 1
    if pos_p==0 or pos_g==0 or neg_p==0 or neg_g==0: return 0.0
    pos_m, neg_m = 0, 0
    for p,g in zip(y_pred, y_gold):
        if p==g:
            if p == 1: pos_m += 1
            elif p == 0: neg_m += 1
    pos_prec, pos_reca = float(pos_m) / pos_p, float(pos_m) / pos_g
    neg_prec, neg_reca = float(neg_m) / neg_p, float(neg_m) / neg_g
    if pos_m == 0 or neg_m == 0: return 0.0
    pos_f1, neg_f1 = 2*pos_prec*pos_reca / (pos_prec+pos_reca), 2*neg_prec*neg_reca / (neg_prec+neg_reca)
    return (pos_f1+neg_f1)/2.0


def get_idx_from_sent(words, word_idx_map, max_l=50):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, user_idx_map, max_l=50):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test13, test14, test15 = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["words"], word_idx_map, max_l)
        sent.append(user_idx_map[rev["uid"]])
        sent.append(rev["y"])
        if rev["split"]==0:
            train.append(sent)
        elif rev["split"]==1:
            dev.append(sent)
        elif rev["split"]==2:
            test13.append(sent)
        elif rev["split"]==3:
            test14.append(sent)
        elif rev["split"]==4:
            test15.append(sent)
    train = np.array(train,dtype="int32")
    dev = np.array(dev,dtype="int32")
    test13 = np.array(test13,dtype="int32")
    test14 = np.array(test14,dtype="int32")
    test15 = np.array(test15,dtype="int32")
    return train, dev, test13, test14, test15
 

if __name__=="__main__":
    np.random.seed(1234)

    #############################
    # Best hyper-parameters     
    #############################
    text_dim = 100    # dimension of sentence representation
    batch_size = 10   # size of mini batches
    n_epochs = 5      # number of training epochs
    ##############################

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    fname, uefname = sys.argv[1:] # dataset and author embedding files

    logger.info("loading data...")
    x = cPickle.load(open(fname,"rb"))
    revs, wordvecs, user_vocab, max_l = x[0], x[1], x[2], x[3]
    logger.info("data loaded!")

    # user embeddings
    uservecs = WordVecs(uefname, user_vocab, binary=0, random=0)
    uservecs.word_idx_map['userID'] = 0

    # train/val/test results
    datasets = make_idx_data(revs, wordvecs.word_idx_map, uservecs.word_idx_map, max_l=max_l)
    train_conv_net(datasets, wordvecs.W, uservecs.W, text_dim=text_dim, batch_size=batch_size, n_epochs=n_epochs)

    logger.info("end logging")
