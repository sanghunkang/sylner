#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce
import pickle as cPickle
import time

# Import external packages
import numpy as np
import tensorflow as tf

from models import SylnerModel

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("fpath_data_train", "../../dev-data/pickle/test_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_string("fpath_data_eval", "../../dev-data/pickle/validation_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_integer("num_class", 2, "Learning rate, epsilon")
tf.flags.DEFINE_integer("batch_size", 128, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning rate, epsilon")


def read_data():

	return data

class Input_sylner():
	"""The input data."""
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size
		self.num_steps = num_steps
		# self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

		self.inputs_seq_char = reader.some_function()#tf.placeholder(tf.int32, shape=[None, 3, None])
		self.inputs_index_target = reader.some_function()#tf.placeholder(tf.int32, shape=[2]) # Start-index and end-index
		self.labels_train = reader.some_function()#tf.placeholder(tf.int32, shape=[None, num_class])



# BUILDING THE COMPUTATIONAL GRAPH
graph = tf.Graph()
with graph.as_default():
	sylnerModel = SylnerModel()

	# Hyperparameters
	learning_rate = FLAGS.learning_rate
	display_step = 10

	# tf Graph input
	num_class = FLAGS.num_class # Normal or Abnormal
	dim_embed = 12

	# Placeholders
	count_step = tf.Variable(0, name="count_step")
	embedding_cho = tf.Variable(tf.random_uniform([num_char_cho, dim_embed], -1.0, 1.0), name="embeddings_cho")
	embedding_jung = tf.Variable(tf.random_uniform([num_char_jung, dim_embed], -1.0, 1.0), name="embeddings_jung")
	embedding_jong = tf.Variable(tf.random_uniform([num_char_jong, dim_embed], -1.0, 1.0), name="embeddings_jong")

	batch_seq_s = tf.placeholder(tf.int32, shape=[None, max_length, 3])
	batch_target_idx = tf.placeholder(tf.int32, shape=[None, 2]) # Start-index and end-index
	batch_labels = tf.placeholder(tf.int32, shape=[None, num_class]) # One-hot encoding
	
	# Define loss, compute gradients
	pred = sylnerModel.run_model(batch_seq_s, batch_target_idx)
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=batch_labels)
	cost = tf.reduce_mean(xentropy)
	gradients = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(gradients)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# Write summary for Tensorboard
	tf.summary.scalar('loss', loss)
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
