from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
from networks import Conv_4, ResNet12



FLAGS = flags.FLAGS

class LWML:
    def __init__(self, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.update_lr = FLAGS.update_lr
        
        if FLAGS.backbone == 'Conv4':
            self.net = Conv_4()
        else:
            self.net = ResNet12()

        self.forward = self.net.forward
        self.construct_weights = self.net.construct_weights

        self.loss_func = xent
        self.classification = True
        self.dim_hidden = 32

        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

        alpha_initializer = tf.initializers.random_uniform(minval=FLAGS.update_lr * 0.99, maxval=FLAGS.update_lr)
        self.alpha1 = tf.get_variable('alpha1', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)
        self.alpha2 = tf.get_variable('alpha2', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)
        self.alpha3 = tf.get_variable('alpha3', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)
        self.alpha4 = tf.get_variable('alpha4', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)
        self.alpha5 = tf.get_variable('alpha5', shape=[1, ], dtype=tf.float32, initializer=alpha_initializer)

        shape = [FLAGS.meta_batch_size, None, 84, 84, 3]
        self.inputa = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.meta_batch_size, None, 84, 84, 3]
        self.inputb = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.meta_batch_size, None, FLAGS.num_classes]
        self.labela = tf.placeholder(tf.float32, shape=shape)
        shape = [FLAGS.meta_batch_size, None, FLAGS.num_classes]
        self.labelb = tf.placeholder(tf.float32, shape=shape)
        
        
    def construct_model(self, input_tensors=None, num_updates=1, train=True):
        # a: training data for inner gradient, b: test data for meta gradient
        self.net.train_flag = train

        with tf.variable_scope('', reuse=tf.AUTO_REUSE) as training_scope:
        #with tf.variable_scope('model', reuse=None) as training_scope:
            # alpha_vectors = []
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights1, weights2, weights3, weights4, weights5 = self.weights1, self.weights2, self.weights3, self.weights4, self.weights5
                weights = self.weights
            else:
                # Define the weights
                weights1, weights2, weights3, weights4, weights5 = self.construct_weights()
                self.weights1, self.weights2, self.weights3, self.weights4, self.weights5 = weights1, weights2, weights3, weights4, weights5
                self.weights = {}
                self.weights.update(self.weights1)
                self.weights.update(self.weights2)
                self.weights.update(self.weights3)
                self.weights.update(self.weights4)
                self.weights.update(self.weights5)
                weights = self.weights

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                if reuse == False:
                    return None
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))

                for key in weights.keys():
                    if key in weights1.keys():
                        gradients[key] = gradients[key]*self.alpha1
                    elif key in weights2.keys():
                        gradients[key] = gradients[key]*self.alpha2
                    elif key in weights3.keys():
                        gradients[key] = gradients[key]*self.alpha3
                    elif key in weights4.keys():
                        gradients[key] = gradients[key]*self.alpha4
                    elif key in weights5.keys():
                        gradients[key] = gradients[key]*self.alpha5
                    else:
                         pass
                fast_weights = dict(zip(weights.keys(), [weights[key] - gradients[key] for key in weights.keys()]))

                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    for key in weights.keys():
                        if key in weights1.keys():
                            gradients[key] = gradients[key] * self.alpha1
                        elif key in weights2.keys():
                            gradients[key] = gradients[key] * self.alpha2
                        elif key in weights3.keys():
                            gradients[key] = gradients[key] * self.alpha3
                        elif key in weights4.keys():
                            gradients[key] = gradients[key] * self.alpha4
                        elif key in weights5.keys():
                            gradients[key] = gradients[key] * self.alpha5
                        else:
                            pass
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - gradients[key] for key in fast_weights.keys()]))

                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                for j in range(num_updates):
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result


        ## Performance & Optimization
        if train:
            self.total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs

            self.total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            optimizer = tf.train.AdamOptimizer(self.meta_lr)

            weight_l_loss0 = 0
            if FLAGS.l2_alpha > 0:
                for key, array in self.weights.items():
                    weight_l_loss0 += tf.reduce_sum(tf.square(array)) * FLAGS.l2_alpha
            if FLAGS.l1_alpha > 0:
                for key, array in self.weights.items():
                    weight_l_loss0 += tf.reduce_sum(tf.abs(array)) * FLAGS.l1_alpha

            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates - 1] + weight_l_loss0)
            gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
            self.metatrain_op = optimizer.apply_gradients(gvs)

        else:
            self.metaval_total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]


