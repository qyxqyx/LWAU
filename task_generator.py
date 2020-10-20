""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
import os
import pwd
import cv2
import math

FLAGS = flags.FLAGS


def random_crop(img, scale=(0.6, 1.0), ratio=(3. / 4., 4. / 3.)):
    shape = img.shape
    area = shape[0] * shape[1]
    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= shape[1] and h <= shape[0]:
            i = random.randint(0, shape[0] - h)
            j = random.randint(0, shape[1] - w)

            croped_img = img[i:i+h,j:j+w,:]
            croped_img = cv2.resize(croped_img, (84, 84))

            return croped_img

    # Fallback
    w = min(shape[0], shape[1])
    i = (shape[0] - w) // 2
    j = (shape[1] - w) // 2
    croped_img = img[i:i + w, j:j + w, :]
    croped_img = cv2.resize(croped_img, (84, 84))

    return croped_img


def random_flip(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    return img



class TaskGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (84, 84))
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = self.num_classes

        if pwd.getpwuid(os.getuid())[0] == 'qyx':
            root = '/home/qyx/ML'
        else:
            root = '/home/yanlb/code/qyx'

        metatrain_folder = config.get('metatrain_folder', root + '/miniimagenet/train')
        if FLAGS.test_set:
            metaval_folder = config.get('metaval_folder', root + '/miniimagenet/test')
        else:
            metaval_folder = config.get('metaval_folder', root + '/miniimagenet/val')

        metatrain_folders = [os.path.join(metatrain_folder, label) \
            for label in os.listdir(metatrain_folder) \
            if os.path.isdir(os.path.join(metatrain_folder, label)) \
            ]
        metaval_folders = [os.path.join(metaval_folder, label) \
            for label in os.listdir(metaval_folder) \
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ]
        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders
        self.rotations = config.get('rotations', [0])

        self.num_total_train_batches = FLAGS.num_train_tasks
        self.num_total_val_batches = 600
        self.pointer = 0

        self.store_data_per_task(train=False)


    def store_data_per_task(self, train=True):
        folders = self.metaval_character_folders
        self.val_tasks_data_classes = []
        for i in range(self.num_total_val_batches):
            task_folders = random.sample(folders, self.num_classes)
            random.shuffle(task_folders)
            support, query = get_images(task_folders, nb_samples=self.num_samples_per_class)
            data_class_task = Files_per_task(support, query, i)
            self.val_tasks_data_classes.append(data_class_task)


    def read_data_per_tesk(self, task_index, train=True):
        if train:
            folders = self.metatrain_character_folders
            task_folders = random.sample(folders, self.num_classes)
            random.shuffle(task_folders)
            train_files, test_files = get_images(task_folders, nb_samples=self.num_samples_per_class)

        else:
            task_class = self.val_tasks_data_classes[task_index]
            train_files = task_class.support
            test_files = task_class.query

        random.shuffle(train_files)
        random.shuffle(test_files)

        image_list = []
        label_list = []
        for image_and_label in train_files:
            image = cv2.imread(image_and_label[1])
            
            if train and FLAGS.data_aug and (FLAGS.backbone=='ResNet12'):
                if random.random() < 0.6:
                    image = random_crop(image)
                image = random_flip(image)

            im2 = image.astype(np.float32) / 256
            im2 = im2.reshape(84, 84, 3)
            image_list.append(im2[np.newaxis, :])

            # label = np.array(image_and_label[0]).reshape((1,))
            label = image_and_label[0]
            label_list.append(label)

        task_train_ims = np.concatenate(image_list, axis=0)
        # task_train_lbls = np.concatenate(label_list, axis=0)
        task_train_lbls = np.array(label_list)
        task_train_lbls = make_one_hot(task_train_lbls, self.num_classes)

        image_list = []
        label_list = []
        for image_and_label in test_files:
            image = cv2.imread(image_and_label[1])

            if train and FLAGS.data_aug and (FLAGS.backbone=='ResNet12'):
                if random.random() < 0.6:
                    image = random_crop(image)
                image = random_flip(image)

            im2 = image.astype(np.float32) / 256
            im2 = im2.reshape(84, 84, 3)
            image_list.append(im2[np.newaxis, :])

            # label = np.array(image_and_label[0]).reshape((1,))
            label = image_and_label[0]
            label_list.append(label)

        task_test_ims = np.concatenate(image_list, axis=0)
        # task_test_lbls = np.concatenate(label_list, axis=0)
        task_test_lbls = np.array(label_list)
        task_test_lbls = make_one_hot(task_test_lbls, self.num_classes)

        return task_train_ims, task_train_lbls, task_test_ims, task_test_lbls


    def get_data_n_tasks(self, meta_batch_size, train=True):
        if train:
            task_indexes = list(range(meta_batch_size))
        else:
            task_indexes = list(range(self.pointer, self.pointer + meta_batch_size))
            if self.pointer + meta_batch_size >= self.num_total_val_batches:
                self.pointer = 0
            else:
                self.pointer += meta_batch_size

        train_ims = []
        train_lbls = []

        test_ims = []
        test_lbls = []

        for task_index in task_indexes:
            task_train_ims, task_train_lbls, task_test_ims, task_test_lbls = \
                self.read_data_per_tesk(task_index, train)
            train_ims.append(task_train_ims[np.newaxis, :])
            train_lbls.append(task_train_lbls[np.newaxis, :])
            test_ims.append(task_test_ims[np.newaxis, :])
            test_lbls.append(task_test_lbls[np.newaxis, :])

        meta_train_ims = np.concatenate(train_ims, axis=0)
        meta_train_lbls = np.concatenate(train_lbls, axis=0)
        meta_test_ims = np.concatenate(test_ims, axis=0)
        meta_test_lbls = np.concatenate(test_lbls, axis=0)

        return meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls


def make_one_hot(data, classes):
    return (np.arange(classes) == data[:, None]).astype(np.integer)


class Files_per_task(object):
    def __init__(self, support, query, task_index):
        self.support = support
        self.query = query
        self.task_index = task_index

    def read_images(self):
        pass


