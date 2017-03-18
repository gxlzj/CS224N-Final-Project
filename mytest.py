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



# x = Input(shape=(32,))
# layer = Dense(32)
# layer2 = Dense(1, trainable=False)
# h = layer(x)
# y = layer2(h)

# layer.trainable = False
# # layer2.trainable = False
# frozen_model = Model(x, y)
# # in the model below, the weights of `layer` will not be updated during training
# frozen_model.compile(optimizer='rmsprop', loss='mse')

# layer.trainable = True
# # layer2.trainable = False
# trainable_model = Model(x, y)
# # with this model the weights of the layer will be updated during training
# # (which will also affect the above model since it uses the same layer instance)
# trainable_model.compile(optimizer='rmsprop', loss='mse')

# import numpy as np
# data = np.random.random((1000, 32))
# labels = np.random.randint(2, size=(1000, 1))

# frozen_model.fit(data, labels, epochs=1)  # this does NOT update the weights of `layer`
# print '--------'
# trainable_model.fit(data, labels, epochs=1)  # this updates the weights of `layer`



x = Input(shape=(32,))
layer = Dense(1)
y = layer(x)

layer.trainable = False
frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

import numpy as np
data = np.random.random((1000, 32))
labels = np.random.randint(2, size=(1000, 1))

print 'frozen train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
frozen_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'

print 'trainable train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
trainable_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'

print 'frozen train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
frozen_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'

print 'trainable train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
trainable_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'

print 'frozen train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
frozen_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'

print 'trainable train'
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print 'after'
trainable_model.fit(data, labels, epochs=1, verbose = False)  # this does NOT update the weights of `layer`
pred_f = frozen_model.predict(data)
print "frozen model error:" + str(np.sum((labels - pred_f) * (labels - pred_f)))
pred_t = trainable_model.predict(data)
print "trainable model error:" + str(np.sum((labels - pred_t) * (labels - pred_t)))
print '--------'