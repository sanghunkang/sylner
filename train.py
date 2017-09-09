#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce
import pickle as cPickle
import time

# Import external packages
import numpy as np
import tensorflow as tf


from models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("fpath_data_train", "../../dev-data/pickle/test_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_string("fpath_data_eval", "../../dev-data/pickle/validation_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_integer("num_class", 2, "Learning rate, epsilon")
tf.flags.DEFINE_integer("batch_size", 128, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning rate, epsilon")


# BUILDING THE COMPUTATIONAL GRAPH
graph = tf.Graph()
with graph.as_default():
	# Hyperparameters
	learning_rate = FLAGS.learning_rate
	display_step = 10

	# tf Graph input
	len_input = 224*224*3
	num_class = FLAGS.num_class # Normal or Abnormal

	# Placeholders

	count_step = tf.Variable(0, name="count_step")
	
	inputs_train = tf.placeholder(tf.int32, shape=[batch_size])
	labels_train = tf.placeholder(tf.int32, shape=[batch_size, 1])
	
	test_dataset = tf.constant(test_examples, dtype=tf.int32)

	# Write summary for Tensorboard
	tf.summary.scalar('loss', loss)
	tf.summary.histogram('W_nce', b_nce)
	tf.summary.histogram('b_nce', b_nce)
	merged = tf.summary.merge_all()

	# Define saver to save/restore trained result
	saver = tf.train.Saver()

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):

	# Define saver
	saver = tf.train.Saver()

	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(config=config) as sess:
		num_steps = FLAGS.num_steps
		batch_size = FLAGS.batch_size

if __name__ == "__main__":
	tf.app.run()
