import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import scipy.spatial as sp
from sklearn.utils import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

train_h5 = './data/train_256.h5'
test_h5 = './data/test_256.h5'

def placeholder_inputs(batch_size, num_point):
	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
	labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
	return pointclouds_pl, labels_pl

def read_train_dataset(k=256, skip_rate=1, rt=None):
	ds = h5py.File(train_h5, 'r')
	data, label = ds['points'][1:, :k, :3], ds['label'][1:, :3]
	data, label = data[::skip_rate], label[::skip_rate]
	ds.close()
	return data, label

def read_test_dataset(k=256, skip_rate=1):
	ds = h5py.File(test_h5, 'r')
	data, label = ds['points'][1:, :k, :3], ds['label'][1:, :3]
	data, label = data[::skip_rate], label[::skip_rate]
	ds.close()
	return data,label

def read_eval_dataset(k=256):
	ds = h5py.File(eval_h5, 'r')
	data, label, real = ds['points'][1:, :k], ds['label'][1:], ds['data'][1:]
	ds.close()
	return data,label,real


def shuffle_data_and_label(data, label):
    print 'data ', len(data)
    print 'label ', len(label)
    assert (len(data) == len(label))
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    return data[randomize], label[randomize]

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx], labels[idx]

def search(point, data, label, k=150):
    data_tree, label_tree = sp.cKDTree(data), sp.cKDTree(label)
    _, idx = data_tree.query(point, k)
    neighbor = [data[i]-point for i in idx]
    _, one = label_tree.query(point, 1)
    return neighbor, label[one]-point

def neighborhood_search(idx, data, label, k = 150):
    """
    :param data: the recover data
    :param label: the nearst neighorhood in origin data
    :param k: the neighorhodd size
    :return: the knn neighborhood and the nearst point
    """
    label_kdtree = sp.cKDTree(label)
    data_kdtree = sp.cKDTree(data)
    _ , neighbor = data_kdtree.query(data[idx], k)
    neighborhood = [data[i] for i in neighbor]
    _ , point = label_kdtree.query(data[idx], 1)
    point = label[point]
    return neighborhood, point


def scale_point_cloud(batch_data, batch_label):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_label = np.zeros(batch_label.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        scale_num = np.random.uniform() * 5
        rotation_matrix = np.array([[scale_num, 0, 0],
                                    [0, scale_num, 0],
                                    [0, 0, scale_num]])
        shape_pc, shape_label  = batch_data[k, ...], batch_label[k]
        rotated_data[k, ...], rotated_label[k] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix), np.dot(shape_label, rotation_matrix)
        
    return rotated_data, rotated_label


def rotate_point_cloud_in_plane(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    #rotated_label = np.zeros(batch_label.shape, dtype=np.float32)
    #rotation_angle = np.random.uniform() * 2 * np.pi
    #cosval = np.cos(rotation_angle)
    #sinval = np.sin(rotation_angle)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data



def rotate_point_cloud(batch_data, batch_label):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_label = np.zeros(batch_label.shape, dtype=np.float32)
    #rotation_angle = np.random.uniform() * 2 * np.pi
    #cosval = np.cos(rotation_angle)
    #sinval = np.sin(rotation_angle)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc, shape_label = batch_data[k, ...], batch_label[k]
        rotated_data[k, ...], rotated_label[k] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix), np.dot(shape_label, rotation_matrix)
    return rotated_data, rotated_label


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
