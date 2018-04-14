import cv2
import os
import glob
#from sklearn.utils import shuffle
import numpy as np

class_map_num = {
  "B" : 0,
  "C" : 1,
  "J" : 2,
  "L" : 3,
  "W" : 4,
  "p" : 1,
  "b" : 0,
  "c" : 1,
  "j" : 2,
  "l" : 3,
  "w" : 4,
}

class_list = [ "Brushing", "Cutting", "Jumping", "Lunges", "Wall" ]


def load_own_fold_img(own_path, length):
    images = []
    labels = []
    img_names = []
    cls = []

    #index = fold - 1 (folds = {1,2,3,4,5})
    #index = fold - 1
    print('Now going to read files from own videos')
    path = os.path.join(own_path, '*g')
    # print("path is=" + path)
    
    files = glob.glob(path)
    for fl in files:
        print(" reading file " + os.path.basename(fl))
        name, ext = os.path.splitext(os.path.basename(fl))
        num = class_map_num.get(name[0:1])
        #Don't read the videos or flow files
        if ext == ".avi" or ext == ".flo":
          continue

        image = cv2.imread(fl)
        #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        # label = np.zeros(length)
        # label[num] = 1.0
        labels.append(num)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
        cls.append(class_list[num])
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def load_own_fold_flow(own_path, length):
    flows = []
    labels = []
    img_names = []
    cls = []

    # print('Going to read training images from fold ' + fold)
    #index = fold - 1 (folds = {1,2,3,4,5})
    # index = fold - 1
    print('Now going to read files from own videos')
    path = own_path
    files = glob.glob(path)
    for fl in files:
        name, ext = os.path.splitext(os.path.basename(fl))
        num = class_map_num.get(name[0:1])
        #Don't read the videos or jpg files
        if ext == ".avi" or ext == ".jpg":
          continue

        flow = cv2.optflow.readOpticalFlow(fl)
        flows.append(flow)


        # image = cv2.imread(fl)
        #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        # image = np.multiply(image, 1.0 / 255.0)
        # images.append(image)
        # label = np.zeros(length)
        # label[num] = 1.0
        labels.append(num)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
        cls.append(class_list[num])
    flows = np.array(flows)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return flows, labels, img_names, cls


def load_train_fold_img(train_path, fold, length):
    images = []
    labels = []
    img_names = []
    cls = []

    #index = fold - 1 (folds = {1,2,3,4,5})
    #index = fold - 1
    print('Now going to read files from fold {}'.format(fold))
    path = os.path.join(train_path, str(fold), '*g')
    # print("path is=" + path)
    files = glob.glob(path)
    for fl in files:
        name, ext = os.path.splitext(os.path.basename(fl))
        num = class_map_num.get(name[2:3])
        #Don't read the videos or flow files
        if ext == ".avi" or ext == ".flo":
          continue

        image = cv2.imread(fl)
        #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        # label = np.zeros(length)
        # label[num] = 1.0
        labels.append(num)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
        cls.append(class_list[num])
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

def load_train_fold_flow(train_path, fold, length):
    flows = []
    labels = []
    img_names = []
    cls = []

    # print('Going to read training images from fold ' + fold)
    #index = fold - 1 (folds = {1,2,3,4,5})
    # index = fold - 1
    print('Now going to read files from fold {}'.format(fold))
    path = os.path.join(train_path, str(fold), '*g')
    files = glob.glob(path)
    for fl in files:
        name, ext = os.path.splitext(os.path.basename(fl))
        num = class_map_num.get(name[2:3])        
        #Don't read the videos or jpg files
        if ext == ".avi" or ext == ".jpg":
          continue

        flow = cv2.optflow.readOpticalFlow(fl)
        flows.append(flow)


        # image = cv2.imread(fl)
        #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        # image = np.multiply(image, 1.0 / 255.0)
        # images.append(image)
        # label = np.zeros(length)
        # label[num] = 1.0
        labels.append(num)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
        cls.append(class_list[num])
    flows = np.array(flows)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return flows, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, own_path, validation_num, read_flow):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # train_images = np.empty(0)
  # train_labels = np.empty(0)
  # train_img_names = np.empty(0)
  # train_cls = np.empty(0)

  images_array = []
  labels_array = []
  img_names_array = []
  cls_array = []


  for i in range(1, 6):
    if i == validation_num:
      continue
    
    if not read_flow:
      images, labels, img_names, cls = load_train_fold_img(train_path, i, 5)
    else:
      images, labels, img_names, cls = load_train_fold_flow(train_path, i, 5)

    images_array.append(images)
    labels_array.append(labels)
    img_names_array.append(img_names)
    cls_array.append(cls)


  train_images = np.concatenate((images_array[0], images_array[1], images_array[2], images_array[3]))
  train_labels = np.concatenate((labels_array[0], labels_array[1], labels_array[2], labels_array[3]))
  train_img_names = np.concatenate((img_names_array[0], img_names_array[1], img_names_array[2], img_names_array[3]))
  train_cls = np.concatenate((cls_array[0], cls_array[1], cls_array[2], cls_array[3]))

  if validation_num != 0:
    
    if not read_flow:
      validation_images, validation_labels, validation_img_names, validation_cls = load_train_fold_img(train_path, validation_num, 5)
    else:
      validation_images, validation_labels, validation_img_names, validation_cls = load_train_fold_flow(train_path, validation_num, 5)

  else:
    if not read_flow:
      validation_images, validation_labels, validation_img_names, validation_cls = load_own_fold_img(own_path, 5)
    else:
      validation_images, validation_labels, validation_img_names, validation_cls = load_own_fold_flow(own_path, 5)

  # images, labels, img_names, cls = load_train_fold_img(train_path, classes, 5)
  #images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  # if isinstance(validation_size, float):
  #   validation_size = int(validation_size * images.shape[0])

  # validation_images = images[:validation_size]
  # validation_labels = labels[:validation_size]
  # validation_img_names = img_names[:validation_size]
  # validation_cls = cls[:validation_size]

  # train_images = images[validation_size:]
  # train_labels = labels[validation_size:]
  # train_img_names = img_names[validation_size:]
  # train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets


