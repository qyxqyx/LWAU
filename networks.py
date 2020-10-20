import tensorflow as tf
from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
import numpy as np

FLAGS = flags.FLAGS

class Conv_4(object):
    def __init__(self):
        '''
        Conv-4 backbone
        '''
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






class ResNet12(object):
    '''
    resnet12 backbone
    '''
    def __init__(self):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.dim_output = FLAGS.num_classes
        self.img_size = 84
        self.train_flag = True


    def construct_weights(self):
        weights = {}
        weights1, weights2, weights3, weights4, weights5 = {}, {}, {}, {}, {}
        k = 3
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)

        for i in range(4):
            block_name = str(i+1)
            for j in ['a', 'b', 'c']:
                var_name = block_name+'/'+j+'/conv/'+'kernel'
                var_filters = FLAGS.base_num_filters * np.power(2, i)
                if i == 0 and j == 'a':
                    input_filters = 3
                elif j == 'a':
                    input_filters = var_filters / 2
                else:
                    input_filters = var_filters
                var_shape = [3, 3, input_filters, var_filters]
                weights[var_name] = tf.get_variable(var_name,
                                                    var_shape,
                                                    initializer=conv_initializer,
                                                    dtype=dtype)


                var_name = block_name + '/' + j + '/conv/' + 'bias'
                var_shape = [var_filters, ]
                weights[var_name] = tf.get_variable(var_name,
                                                    var_shape,
                                                    initializer=fc_initializer,
                                                    dtype=dtype)

            var_name = block_name + '/shortcut/conv/kernel'
            var_filters = FLAGS.base_num_filters * np.power(2, i)
            if i == 0:
                input_filters = 3
            else:
                input_filters = var_filters / 2

            var_shape = [1, 1, input_filters, var_filters]
            weights[var_name] = tf.get_variable(var_name,
                                                var_shape,
                                                initializer=conv_initializer,
                                                dtype=dtype)

            var_name = block_name + '/shortcut/conv/bias'
            var_shape = [var_filters, ]
            weights[var_name] = tf.get_variable(var_name,
                                                var_shape,
                                                initializer=fc_initializer,
                                                dtype=dtype)


        weights['5/kernel'] = tf.get_variable('dense/kernel',
                                                  [FLAGS.base_num_filters * np.power(2, 3), self.dim_output],
                                                  initializer=fc_initializer)
        weights['5/bias'] = tf.get_variable('dense/bias', [self.dim_output],
                                                initializer=fc_initializer)

        for key, var in weights.items():
            if key[0] == '1':
                weights1[key] = var
            elif key[0] == '2':
                weights2[key] = var
            elif key[0] == '3':
                weights3[key] = var
            elif key[0] == '4':
                weights4[key] = var
            else:
                weights5[key] = var


        return weights1, weights2, weights3, weights4, weights5


    def forward(self, inp, weights, reuse=False, scope=''):

        feature = tf.reshape(inp, [-1, 84, 84, 3])

        for i in range(4):
            block_name = str(i + 1)

            kernel_name = block_name + '/shortcut/conv/kernel'
            bias_name = block_name + '/shortcut/conv/bias'

            shortcut = tf.nn.convolution(feature, weights[kernel_name], padding='SAME', strides=[1,1]) + weights[bias_name]

            for j in ['a', 'b']:
                kernel_name = block_name + '/' + j + '/conv/' + 'kernel'
                bias_name = block_name + '/' + j + '/conv/' + 'bias'

                feature = tf.nn.convolution(feature, weights[kernel_name], padding='SAME', strides=[1, 1]) + \
                           weights[bias_name]


                feature = tf.layers.batch_normalization(feature, training=True,
                                                        name=block_name + '/' + j + '/bn',
                                                        reuse=reuse)
                feature = tf.nn.relu(feature)

            kernel_name = block_name + '/c/conv/' + 'kernel'
            bias_name = block_name + '/c/conv/' + 'bias'

            feature = tf.nn.convolution(feature, weights[kernel_name], padding='SAME', strides=[1, 1]) + \
                      weights[bias_name]

            feature = feature + shortcut


            feature = tf.layers.batch_normalization(feature, training=True,
                                                    name=block_name + '/' + j + '/bn',
                                                    reuse=reuse)
            feature = tf.nn.relu(feature)

            feature = tf.layers.max_pooling2d(feature, [2, 2], [2, 2], 'same')

        feature = tf.reduce_mean(feature, axis=[1, 2])

        if FLAGS.dropout_rate > 0:
            feature = tf.layers.dropout(feature, FLAGS.dropout_rate, training=self.train_flag, seed=1)

        fc1 = tf.matmul(feature, weights['5/kernel']) + weights['5/bias']
        return fc1









