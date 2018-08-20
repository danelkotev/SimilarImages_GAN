import numpy as np
import os
import sys
import csv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


class DataSet:
     def __init__(self, data, label):
         self.data = data
         self.label = label


class DataSets:
    def __init__(self, data_sets, labels):
        self.data_sets = data_sets
        self.labels = labels


def merge(images, size):
    """
    Generates a merged image from all the input images.
    :param images: Images to be merged.
    :param size: [number of rows, number of columns]
    :return: Merged image.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[int(j) * h:int(j) * h + h, int(i) * w:int(i) * w + w, :] = image

    return img


def load_dataset(images_path, labels_path, image_size=64):
    """
    :param images_path: examples path
    :param labels_path: labels path as CSV with one column, no headers, number of lines should be as number of images
    :param image_size: image size
    :return:
    """
    all_dirs = filter(lambda x: not (x.startswith('.')) ,os.listdir(images_path))
    image_dirs = \
        [i + '/' + j for i in all_dirs for j in os.listdir(images_path + '/' + i) if j.endswith(".jpg") or j.endswith(".jpeg") or j.endswith(".png")]
    number_of_images = len(image_dirs)
    images=[]
    datasets = []
    print("images are being loaded...")
    for c, i in enumerate(image_dirs):
        im = Image.open(images_path + '/' + i)
        im = im.resize([image_size, image_size], Image.ANTIALIAS)
        image = np.array(im)
        #np.array(ImageOps.fit(im,(image_size, image_size), Image.ANTIALIAS)) /127.5 - 1.
        #np.array(ImageOps.fit(,
        #                               (image_size, image_size), Image.ANTIALIAS))  # /127.5 - 1.
        sys.stdout.write("\r Loading : {}/{}".format(c + 1, number_of_images))
        images.append(image)
        dataset = DataSet(image, 1)
        datasets.append(dataset)
    # with open(labels_path) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     index=0
    #     for row in csv_reader:
    #         dataset=DataSet(images[index], 1) #float(row[0]))
    #         datasets.append(dataset)
    #         index=index+1
    return datasets


def next_batch(data, batch_size, image_size):
    """
    Returns a random chosen batch from the data array.
    :param data: numpy array consisting the entire dataset
    :param batch_size: should I even explain.
    :return: [batch_size, default image size, default image size, 3]
    """
    batch = np.random.permutation(data)[:batch_size]
    data = list(map(lambda x: x.data,batch))
    labels = list(map(lambda x: x.label, batch))
    datasets = DataSets(np.reshape(data, [-1, image_size, image_size, 3]), labels)
    return datasets