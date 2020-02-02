import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    return pointclouds_pl, labels_pl


def block(input, count, is_training, bn_decay, wd=0.0, dp_rate=0.9):
	
	if count != 0:
		shortcut = input
	
	if count == 0:
		C = 360
	else:
		C = input.get_shape()[-1]

	with tf.name_scope('block%d' % (count)) as _scope:
		net = tf_util.conv2d(input, C , [3,3], padding='SAME', stride=[1,1], weight_decay=wd,
							 bn=True, is_training=is_training, bn_decay=bn_decay,
							 scope=('%dres_conv1' % (count)), activation_fn=tf.nn.relu)

		net = tf_util.conv2d(net, C , [3,3], padding='SAME', stride=[1,1], weight_decay=wd,
							 bn=True, is_training=is_training, bn_decay=bn_decay,
							 scope=('%dres_conv2' % (count)), activation_fn=tf.nn.relu)


	#if count != 0:
		#net = shortcut + net
	
	net = tf.nn.relu(net)
	return net


def out(net, batch_size, h_size=9, data_form='bhwc'):
	net = tf.reshape(net, [batch_size, -1, h_size])
	half = net.shape[1] / 2
	net1 = net[:, :half, :,]
	net2 = net[:, half:, :,]
	if data_form == 'bchw':
		net = tf.einsum('abc,abf->abcf', net1, net2)
	else:
		net = tf.einsum('abc,abf->acfb', net1, net2)
	return net
	
def get_model(point_cloud, is_training, bn_decay=None, block_size=2, wd=0.0, h_size=9):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1,3], weight_decay=wd,
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], weight_decay=wd,
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],weight_decay=wd,
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 16*h_size**2, [1,1],weight_decay=wd,
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
	                         padding='VALID', scope='max_pool')

    net = out(net, batch_size, h_size)
    
    for i in range(block_size):
        net = block(net, i, is_training, bn_decay, wd)


    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc2',weight_decay=wd)

    return net, end_points


def get_loss(pred, label, end_points, is_training):
    """ pred: B*NUM_CLASSES,
        label: B, """
    reg_loss = tf.cond(is_training, lambda: tf.add_n(tf.get_collection("losses")), lambda: tf.Variable(0.0))    
    loss = tf.losses.mean_squared_error(pred, label)
    return loss + reg_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
