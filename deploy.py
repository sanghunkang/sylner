#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, time

# Import external packages
import numpy as np
import tensorflow as tf

from models import SylnerModel
import utils, hangulvars 

BASE_CODE = hangulvars.BASE_CODE
CHOSUNG = hangulvars.CHOSUNG
JUNGSUNG = hangulvars.JUNGSUNG
NONHANGUL_LIST = hangulvars.NONHANGUL_LIST
CHOSUNG_LIST = hangulvars.CHOSUNG_LIST
JUNGSUNG_LIST = hangulvars.JUNGSUNG_LIST
JONGSUNG_LIST = hangulvars.JONGSUNG_LIST

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("fpath_data", "../../dev-data/sylner/base_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_string("logdir", "./logs", "The directory to save the model files in.")
tf.flags.DEFINE_integer("num_class", 5, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")

# Deactivate minor warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def feed_data(data, batch_size, num_class):
	# np.random.shuffle(data)
	batch = data
	len_seq = (data.shape[1] - num_class)//4

	return {batch_seq_s: np.reshape(batch[:, :len_seq*3],[-1,len_seq,3]),
			batch_clipper: batch[:, len_seq*3:len_seq*4],
			batch_labels: batch[:, len_seq*4:]}

# BUILDING THE COMPUTATIONAL GRAPH
graph = tf.Graph()
with graph.as_default():
	sylnerModel = SylnerModel()

	# Hyperparameters

	# tf Graph input
	num_class = FLAGS.num_class # Normal or Abnormal

	# Placeholders
	count_step = tf.Variable(0, name="count_step")
	
	max_length = 341 # Temporary hard coding

	batch_seq_s = tf.placeholder(tf.int32, shape=[None, max_length, 3])
	batch_clipper = tf.placeholder(tf.float32, shape=[None, max_length])
	batch_labels = tf.placeholder(tf.int32, shape=[None, num_class]) # One-hot encoding
	
	# Define loss, compute gradients
	pred = sylnerModel.run_model(batch_seq_s, batch_clipper)
	
	# Update Weights
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=batch_labels)

	# Compute Accuracy
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(batch_labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	saver = tf.train.Saver()

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):
	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(graph=graph, config=config) as sess:
		tf.global_variables_initializer().run()

		num_class = FLAGS.num_class
		# data = utils.nikl_to_nparray("../../dev-data/sylner/2016klpNER.base_train")
		data, arr_rec_str = utils.nikl_to_nparray("../../dev-data/sylner/2016klpNER.base_train")
		
		# Load Checkpoint Data
		saver.restore(sess, './{0}/checkpoint.ckpt'.format(FLAGS.logdir))

		accu, pred_print, labels_print = sess.run([accuracy, pred, batch_labels], feed_dict=feed_data(data, data.shape[0], num_class))
		pp, lp = np.argmax(pred_print, axis=1), np.argmax(labels_print, axis=1)
		# print(np.hstack([pred_print, labels_print]))
		
		for i in range(data.shape[0]):
			print(arr_rec_str[i], pp[i], lp[i])

		for tag in range(5):
			count_precision = 0
			base_precision = 0
			for i in range(data.shape[0]):
				if pp[i] == tag:
					base_precision += 1
					if pp[i] == lp[i]: count_precision += 1
			precision = count_precision/base_precision
			
			count_recall = 0
			base_recall = 0
			for i in range(data.shape[0]):
				if lp[i] == tag:
					base_recall += 1
					if pp[i] == lp[i]: count_recall += 1
			recall = count_recall/base_recall
			

			fscore = 2*precision*recall/(precision+recall)
			print("Precision{0}= {1:.5f}".format(tag, precision))
			print("Recall{0}= {1:.5f}".format(tag, recall))
			print("F-score{0}= {1:.5f}".format(tag, fscore))
			print()
		
		print("Accuracy= {0:.5f}".format(accu))
		


if __name__ == "__main__":
	tf.app.run()
