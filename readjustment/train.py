import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import provider
import threading
from sklearn.utils import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='', help='Model for traning')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=256, help='Point Number [128/256] [default: 256]')
parser.add_argument('--max_epoch', type=int, default=1500, help='Epoch to run [default: 1500]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 128]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--load_model', type=bool, default=False, help='Load has been trained model[default:false]')
parser.add_argument('--model_path', default='', help='The Model path[default:'']')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOAD_MODEL = FLAGS.load_model
MODEL_PATH = FLAGS.model_path

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__ ,LOG_DIR)) # bkp of train procedure
os.system('cp provider.py %s' %(LOG_DIR))
os.system('cp %s %s' %('sub.sh' , LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP / 30)
BN_DECAY_CLIP = 0.99
MIN_LEARNING_RATE = 0.00001


TRAIN_DATA, TRAIN_LABEL = provider.read_train_dataset(NUM_POINT)
TEST_DATA, TEST_LABEL = provider.read_test_dataset(NUM_POINT)
print TEST_DATA.shape, TRAIN_DATA.shape
sys.stdout.flush()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, MIN_LEARNING_RATE) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch * NUM_POINT,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    bn_decay = BN_DECAY_CLIP
    return bn_decay



def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = provider.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points, is_training_pl)
            #tf.summary.scalar('reg loss', reg_loss)


            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=3)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph) 
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'learning_rate': learning_rate,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        min_loss = 0x7fffffff

        if LOAD_MODEL:
            ckpt = tf.train.get_checkpoint_state(os.path.join(BASE_DIR, MODEL_PATH))
            if ckpt and ckpt.model_checkpoint_path:
                print '==========================', ckpt.model_checkpoint_path
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'after restore: ', sess.run(batch), sess.run(learning_rate)				
                				
            last_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
            for epoch in range(last_epoch+1, last_epoch+MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, TRAIN_DATA, TRAIN_LABEL, train_writer, epoch)
                e_loss = eval_one_epoch(sess, ops, TEST_DATA, TEST_LABEL, test_writer, epoch)

                # Save the variables to disk.
                if e_loss < min_loss:
                    min_loss = e_loss  				
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                    log_string("Model saved in file: %s" % save_path)
        else:
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, TRAIN_DATA, TRAIN_LABEL,train_writer, epoch)
                e_loss = eval_one_epoch(sess, ops, TEST_DATA, TEST_LABEL, test_writer, epoch)

                # Save the variables to disk.
                if (epoch+1) % 100 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                    log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, DATA, LABEL, train_writer, epoch):
	is_training = True
	num_batchs = int(len(DATA) // BATCH_SIZE)
	loss_sum = 0.0  
	loss_list = []
	DATA, LABEL = shuffle(DATA, LABEL)
	print '------------train----------'

	for i in range(num_batchs):
		begin_idx, end_idx = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
		feed_data, feed_label = DATA[begin_idx:end_idx], LABEL[begin_idx:end_idx]
		feed_dict = {ops['pointclouds_pl']:feed_data,
                    ops['labels_pl']: feed_label,
                    ops['is_training_pl']: is_training}
		summary, step, _, loss_val, pred_val, lr = sess.run([ops['merged'], ops['step'],
                        ops['train_op'], ops['loss'],
                        ops['pred'], ops['learning_rate'], ],
                        feed_dict=feed_dict)
		loss_sum += loss_val
		loss_list.append(loss_val)

	summary1 = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
												simple_value=loss_sum / float(num_batchs),)])

	loss_max, loss_min = max(loss_list), min(loss_list)
	print 'train max:%f min:%f' % (loss_max, loss_min)
	train_writer.add_summary(summary, epoch+1)
	train_writer.add_summary(summary1, epoch+1)
	train_writer.flush()
	log_string('mean loss: %f' % (loss_sum / float(num_batchs)))

def eval_one_epoch(sess, ops, DATA, LABEL, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0.0
    loss_list = []
    num_batchs = int(len(DATA) // BATCH_SIZE)
    print '----------eval------------'
    	
    for i in range(num_batchs): 
        begin_idx, end_idx = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        feed_data, feed_label = DATA[begin_idx:end_idx], LABEL[begin_idx:end_idx] 
        feed_dict = {ops['pointclouds_pl']: feed_data,
                         ops['labels_pl']: feed_label,
                         ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, lr = sess.run([ops['merged'], ops['step'],
                    ops['loss'], ops['pred'], ops['learning_rate']], feed_dict=feed_dict)
        loss_sum += loss_val
        loss_list.append(loss_val)

    summary1 = tf.Summary(value=[tf.Summary.Value(tag="eval_loss", 
                                                 simple_value=loss_sum / float(num_batchs),)])
    loss_max, loss_min = max(loss_list), min(loss_list)
    test_writer.add_summary(summary, epoch+1)
    test_writer.add_summary(summary1, epoch+1)
    test_writer.flush()
    log_string('eval mean loss: %f' % (loss_sum / float(num_batchs)))
    return loss_sum / float(num_batchs)


if __name__ == "__main__":
    train()
    LOG_FOUT.close()

