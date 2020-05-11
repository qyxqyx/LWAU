import numpy as np
import random
import tensorflow as tf
from lwml import LWML
from tensorflow.python.platform import flags
import os
from task_generator import TaskGenerator

FLAGS = flags.FLAGS

flags.DEFINE_integer('metatrain_iterations', 60000, 'number of metatraining iterations.')

# Training options
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification')
flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-training iteration')
flags.DEFINE_float('meta_lr', 0.001, 'the meta learning rate')
flags.DEFINE_float('update_lr', 0.01, 'the inner-update learning rate')
flags.DEFINE_integer('update_batch_size', 1, 'K for K-shot learning.')
flags.DEFINE_integer('num_updates', 5, 'number of inner update steps during training.')
flags.DEFINE_integer('num_train_tasks', 20, 'number of meta training tasks.')
flags.DEFINE_float('l2_alpha', 0.001, 'param of the l2_norm')
flags.DEFINE_float('l1_alpha', 0.001, 'param of the l1_norm')
flags.DEFINE_float('dropout_rate', 0, 'dropout_rate of the FC layer')
flags.DEFINE_integer('base_num_filters', 16, 'number of filters for conv nets.')
flags.DEFINE_integer('test_num_updates', 10, 'number of inner update steps during testing')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/miniimagenet1shot/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_bool('net', False, 'whether use the data saved on the risk disk, or use the data saved on the local disk.')

if FLAGS.train:
    NUM_TEST_POINTS = int(60/FLAGS.meta_batch_size)
else:
    NUM_TEST_POINTS = 60


LEN_MODELS = 50
PRINT_INTERVAL = 5
TEST_PRINT_INTERVAL = PRINT_INTERVAL*6

def train(model, saver, sess, exp_string, resume_itr=0):
    print('Done initializing, starting training.')
    print(exp_string)
    prelosses, postlosses = [], []

    models = {}

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        feed_dict = {model.meta_lr: FLAGS.meta_lr}

        input_tensors = [model.metatrain_op]
        input_tensors.extend([model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
        input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)
        prelosses.append(result[-2])
        postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            metaval_accuracies = []
            for _ in range(NUM_TEST_POINTS):
                feed_dict = {}
                input_tensors = [[model.metaval_total_accuracy1] + model.metaval_total_accuracies2]

                result = sess.run(input_tensors, feed_dict)
                metaval_accuracies.append(result[0])

            metaval_accuracies = np.array(metaval_accuracies)
            means = np.mean(metaval_accuracies, 0)
            stds = np.std(metaval_accuracies, 0)
            ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
            print('----------------------------------------', itr)
            print('Mean validation accuracy:', means)
            print('Mean validation loss:', stds)
            print('Mean validation stddev', ci95)
            print('----------------------------------------', )

            val_postaccs = max(means)
            model_name = FLAGS.logdir + '/' + exp_string + '/model' + str(itr)
            if len(models) >= LEN_MODELS:
                min_acc, min_model = min(zip(models.values(), models.keys()))
                if val_postaccs > min_acc:
                    del models[min_model]
                    models[model_name] = val_postaccs
                    saver.save(sess, model_name)
                    # os.remove(min_model+'.meta')
                    os.remove(min_model + '.data-00000-of-00001')
                    os.remove(min_model + '.index')
                    os.remove(model_name + '.meta')
                else:
                    pass
                max_acc, max_model = max(zip(models.values(), models.keys()))
                print(max_model, ':', max_acc)
            else:
                models[model_name] = val_postaccs
                saver.save(sess, model_name)
                os.remove(model_name + '.meta')

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))


def test(model, sess):
    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    max_acc = 0
    print(NUM_TEST_POINTS)
    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr : 0.0}
        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    for mean_acc in means:
        if mean_acc> max_acc:
            max_acc=mean_acc

    print('Mean validation accuracy:', means)
    print('Mean validation loss:', stds)
    print('Mean validation stddev', ci95)

    return max_acc



def main():
    FLAGS.logdir = 'logs/miniimagenet' + str(FLAGS.update_batch_size) + 'shot/'
    # assert len(lists) >= 5, '列表元素个数小于5'

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1
        orig_update_batch_size = FLAGS.update_batch_size

    task_generator = TaskGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)
    dim_output = task_generator.dim_output
    dim_input = task_generator.dim_input

    if FLAGS.train: # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = task_generator.make_data_tensor()
        inputa = tf.slice(image_tensor, [0,0,0], [-1,FLAGS.num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,FLAGS.num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,FLAGS.num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,FLAGS.num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    random.seed(6)
    image_tensor, label_tensor = task_generator.make_data_tensor(train=False)
    inputa = tf.slice(image_tensor, [0,0,0], [-1,FLAGS.num_classes*FLAGS.update_batch_size, -1])
    inputb = tf.slice(image_tensor, [0,FLAGS.num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,FLAGS.num_classes*FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0,FLAGS.num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = LWML(dim_input, dim_output)
    if FLAGS.train :
        model.construct_model(input_tensors=input_tensors, num_updates=FLAGS.num_updates, train=True)
    model.construct_model(input_tensors=metaval_input_tensors, num_updates=FLAGS.test_num_updates, train=False)

    # model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=0)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size
        FLAGS.update_batch_size = orig_update_batch_size

    exp_string = str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size)
    exp_string += '.nstep_' + str(FLAGS.num_updates) + '.tnstep_' + str(FLAGS.test_num_updates)
    exp_string += '.ubs_' + str(FLAGS.update_batch_size) + '.nts_' + str(FLAGS.num_train_tasks)
    exp_string += '.l1_' + str(FLAGS.l1_alpha) +'.l2_' + str(FLAGS.l2_alpha)
    exp_string += '.lr_' + str(FLAGS.meta_lr) + '.ulr_' + str(FLAGS.update_lr)

    exp_string += '.drop_' + str(FLAGS.dropout_rate) + '.nfs_' + str(FLAGS.base_num_filters)

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, resume_itr)
    else:
        import os
        max_accs = 0
        models = os.listdir(FLAGS.logdir + exp_string)
        model_epochs = []
        for model_file in models:
            if 'model' in model_file and 'index' in model_file:
                i = model_file.find('del')
                j = model_file.find('.')
                model_epoch = model_file[i + 3:j]
                model_epochs.append(int(model_epoch))
        model_epochs.sort()

        max_epoch = 0
        for epoch in model_epochs:
            if epoch > float(FLAGS.metatrain_iterations) / 20:
                model_file = FLAGS.logdir + exp_string + '/model' + str(epoch)
                saver.restore(sess, model_file)
                print("testing model: " + model_file)
                acc = test(model, sess)
                if acc > max_accs:
                    max_accs = acc
                    max_epoch = epoch
                print('----------max_acc:', max_accs, '-----------max_model:', max_epoch)
            else:
                pass


if __name__ == "__main__":
    main()





