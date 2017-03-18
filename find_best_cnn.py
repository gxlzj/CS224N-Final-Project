import sys, cPickle, logging
import numpy as np
np.random.seed(1234)

from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.merge import *
from keras.utils import np_utils
from keras import backend as K
import keras
import tensorflow as tf

from keras_classes import *
from process_data import WordVecs

# logging = logging.getLogger("my_author_model")

def train_conv_net(datasets,                # word indices of train/dev/test tweets
          U,                       # pre-trained word embeddings
          activation_form='tanh',
          text_dim=50,            # dim of sentence vector
          dropout_rate=0.5,
          batch_size=10,           # mini batch size
          L2_penalty=0.001,
          n_epochs=30,
          model_name = 'defaultname.h5',
          ):
  """
  train and evaluate convolutional neural network model for sentiment clasisification with SemEval datasets
  """
  # prepare datasets
  train_set, dev_set, test13_set, test14_set, test15_set = datasets
  train_set_x, dev_set_x, test13_set_x, test14_set_x, test15_set_x = train_set[:,:-1], dev_set[:,:-1], test13_set[:,:-1], test14_set[:,:-1], test15_set[:,:-1]
  train_set_y, dev_set_y, test13_set_y, test14_set_y, test15_set_y = train_set[:,-1], dev_set[:,-1], test13_set[:,-1], test14_set[:,-1], test15_set[:,-1]
  train_set_y_cat = np_utils.to_categorical(train_set_y, 3)

  # build model with keras
  n_tok = len(train_set_x[0])  # num of tokens in a tweet
  vocab_size, emb_dim = U.shape



  #prepare sentence features
  sequence = Input(shape=(n_tok,), dtype='int32')
  word_emb = Embedding(vocab_size, emb_dim, weights=[U], trainable=False, input_length=n_tok)(sequence)

  #prepare convolution layer
  cnn_layer = Convolution1D(text_dim, 2, activation=activation_form)
  sentence_features = Dropout(rate=dropout_rate)(cnn_layer(word_emb))

  #max_pooling #flatten layer
  max_pooling_layer = MaxPooling1D(n_tok - 1)
  flatten_layer = Flatten()

  #pure sentence feature
  final_sentence_features = flatten_layer(max_pooling_layer(sentence_features))

  #softmax layer
  softmax_layer = Dense(3, activation='softmax', kernel_regularizer=l2(L2_penalty))
  pred_cnn = softmax_layer(final_sentence_features)


  #pure cnn model (trainable), no elementwise dot product
  ## set all seeds here!!!
  model_cnn = Model(inputs=sequence, outputs=pred_cnn)
  model_cnn.compile(loss='categorical_crossentropy',
         optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
         metrics=['accuracy'])



  # start training
  bestDev_perf, best13_perf, best14_perf, best15_perf = 0., 0., 0., 0.
  corr13_perf, corr14_perf, corr15_perf = 0., 0., 0.
  best_dev_epoc = 0
  for epo in xrange(n_epochs):
    model_cnn.fit(train_set_x, train_set_y_cat, batch_size=batch_size, epochs=1, verbose=0)

    ypred = model_cnn.predict(train_set_x, batch_size=batch_size, verbose=0).argmax(axis=-1)
    train_perf = avg_fscore(ypred, train_set_y)
    ypred = model_cnn.predict(dev_set_x, batch_size=batch_size, verbose=0).argmax(axis=-1)
    dev_perf = avg_fscore(ypred, dev_set_y)
    ypred = model_cnn.predict(test13_set_x, batch_size=batch_size, verbose=0).argmax(axis=-1)
    test13_perf = avg_fscore(ypred, test13_set_y)
    ypred = model_cnn.predict(test14_set_x, batch_size=batch_size, verbose=0).argmax(axis=-1)
    test14_perf = avg_fscore(ypred, test14_set_y)
    ypred = model_cnn.predict(test15_set_x, batch_size=batch_size, verbose=0).argmax(axis=-1)
    test15_perf = avg_fscore(ypred, test15_set_y)

    if dev_perf >= bestDev_perf:
      bestDev_perf, corr13_perf, corr14_perf, corr15_perf = dev_perf, test13_perf, test14_perf, test15_perf
      model_cnn.save_weights('SavedWeights/'+sys.argv[1]+'/'+model_name+'.h5')
      best_dev_epoc = epo + 1
    best13_perf = max(best13_perf, test13_perf)
    best14_perf = max(best14_perf, test14_perf)
    best15_perf = max(best15_perf, test15_perf)

    logging.info("Epoch: %d Train perf: %.3f Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f Avg %.3f: " %(epo+1, train_perf*100, dev_perf*100, test13_perf*100, test14_perf*100, test15_perf*100, (test13_perf+test14_perf+test15_perf)/3*100))

  logging.info("CORR: Best_Dev_Epoc: %d Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f AVG perf: %.3f" %(best_dev_epoc, bestDev_perf*100, corr13_perf*100, corr14_perf*100, corr15_perf*100, (corr13_perf+corr14_perf+corr15_perf)/3*100))
  # logging.info("BEST: Dev perf: %.3f Test13 perf: %.3f Test14 perf: %.3f Test15 perf: %.3f" %(bestDev_perf*100, best13_perf*100, best14_perf*100, best15_perf*100))


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


def make_idx_data(revs, word_idx_map, max_l=50):
  """
  Transforms sentences into a 2-d matrix.
  """
  train, dev, test13, test14, test15 = [], [], [], [], []
  for rev in revs:
    sent = get_idx_from_sent(rev["words"], word_idx_map, max_l)
    # sent.append(user_idx_map[rev["uid"]])
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
  n_epochs = 30     # number of training epochs, no need to tune this!

  activation_all = {'1':'tanh', '2':'sigmoid', '3':'relu'}
  activation_form = activation_all[sys.argv[1]] ## Choose from  and 
  logging.basicConfig(filename='Find_Best_CNN-' + sys.argv[1] + '.log',filemode='a',format='%(asctime)s:%(message)s', level=logging.INFO)

  # activation_form = 'tanh'
  # logging.basicConfig(filename='find_Best_CNN-1.log',filemode='a',format='%(asctime)s:%(message)s', level=logging.INFO)

  logging.info('begin logging')
  logging.info("loading data...")
  x = cPickle.load(open("data/semeval.pkl", "rb"))    # dataset and word embedding files
  revs, wordvecs, max_l = x[0], x[1], x[3]
  logging.info("data loaded!")

  datasets = make_idx_data(revs, wordvecs.word_idx_map, max_l=max_l)
  logging.info("data tokenized!")
  logging.info('\n\n')

  text_dim_list = [16, 32, 50, 100]
  Dropout_list = [0.1, 0.2, 0.4]
  # batch_size_list = [5, 10, 20]
  batch_size_list = [10]
  L2penalty_NegLog_list = [6, 4, 2]
  
  for text_dim in text_dim_list:
    for dropout_rate in Dropout_list:
      for batch_size in batch_size_list:
        for i in L2penalty_NegLog_list:
          L2_penalty = 10**(-i)
          logging.info('====================new running====================')
          model_name = 'textdim='+str(text_dim) + '_dropout='+str((int)(10 * dropout_rate)) + '_batchsize='+str(batch_size) + '_pnl='+str(i)
          logging.info('current setting:'+model_name)
          train_conv_net(datasets, wordvecs.W, 
            activation_form=activation_form, text_dim=text_dim, dropout_rate=dropout_rate, 
            batch_size=batch_size, L2_penalty = L2_penalty,
            n_epochs=n_epochs, model_name=model_name),
          logging.info('\n\n')

  # logging.info("end logging")
