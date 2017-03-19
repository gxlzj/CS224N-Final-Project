import numpy as np
import tensorflow as tf
#parameters
text_dim = 100
word_emb_size = 600
sentence_length = 40
batch_size = 10

#input data
data_x = np.random.random(size=(batch_size,sentence_length,word_emb_size))
print data_x.shape

#placeholders
word_emb = tf.placeholder(tf.float32,shape=[None,sentence_length,word_emb_size])
keep_prob = tf.placeholder(tf.float32)
l2_penalty = tf.placeholder(tf.float32)
y_train = tf.placeholder(tf.float32,shape=[None,1])

#variables
W_conv = tf.get_variable("W_conv", shape=[2, word_emb_size, text_dim],
           initializer=tf.contrib.layers.xavier_initializer())
conv_layer = tf.nn.conv1d(word_emb, filters=W_conv, stride=1, padding = 'VALID')
dropped_conv_layer = tf.nn.dropout(conv_layer,keep_prob)
activated_conv_layer = tf.nn.tanh(dropped_conv_layer)
pooled_layer = tf.reduce_max(activated_conv_layer, axis=1)
W_softmax = tf.get_variable("W_softmax",
                            shape=[text_dim,3],
                            initializer=tf.contrib.layers.xavier_initializer())
b_softmax = tf.get_variable("b_softmax",
                            shape=[3],
                            initializer=tf.contrib.layers.xavier_initializer())
logits = tf.matmul(pooled_layer, W_softmax) + b_softmax

#loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits))
regularizers = tf.nn.l2_loss(W_softmax) + tf.nn.l2_loss(W_conv)
objective = tf.reduce_mean(loss + l2_penalty * regularizers)

#predictions
pred_y = tf.nn.softmax(logits)

#running
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
a1,a2 = sess.run([logits,pred_y], {word_emb:data_x,keep_prob:0.5})

# print a1
# print '----'
# print a2
# print '----'
# print a3
# print '----'
# print a4
# print '----'
print a1.shape
print a2.shape
# print a3.shape
# print a4.shape
# print a5.shape

