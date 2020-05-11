import tensorflow as tf
from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
import numpy as np

FLAGS = flags.FLAGS

class Conv_4(object):
    def __init__(self):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.dim_output = FLAGS.num_classes
        self.img_size = 84

    def construct_weights(self):
        weights1 = {}
        weights2 = {}
        weights3 = {}
        weights4 = {}
        weights5 = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights1['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights1['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')

        weights2['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights2['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
        weights3['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights3['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
        weights4['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights4['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')

        weights5['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output], initializer=fc_initializer)
        weights5['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')

        return weights1, weights2, weights3, weights4, weights5

    def forward(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']













