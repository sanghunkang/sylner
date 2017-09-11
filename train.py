#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce
import pickle as cPickle
import time

# Import external packages
import numpy as np
import tensorflow as tf


from models import TargetSubtractionModel

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
	# Hyperparameters
	learning_rate = FLAGS.learning_rate
	display_step = 10

	# tf Graph input
	num_class = FLAGS.num_class # Normal or Abnormal

	# Placeholders
	count_step = tf.Variable(0, name="count_step")
	
	"""
	[1,2,1]
	->
	[[1 3 5 6],
	 [1 0 2 7],
	 [1 3 5 6]]
	"""
	embedding_cho = tf.Variable(tf.random_uniform([num_char_cho, DIM_EMBED], -1.0, 1.0),
								name="embeddings_cho")
	embedding_jung = tf.Variable(tf.random_uniform([num_char_jung, DIM_EMBED], -1.0, 1.0),
								name="embeddings_jung")
	embedding_jong = tf.Variable(tf.random_uniform([num_char_jong, DIM_EMBED], -1.0, 1.0),
								name="embeddings_jong")

	inputs_seq_char = tf.placeholder(tf.int32, shape=[None, 3, None])
	inputs_index_target = tf.placeholder(tf.int32, shape=[2]) # Start-index and end-index
	labels_train = tf.placeholder(tf.int32, shape=[None, num_class]) # One-hot encoding
	"""
	"Correctness"
	
	Prediction given index, character, 

	Network to extract context feature
	Network to extract word feature
	"""
	
	# Define loss, compute gradients
	pred, labels = model_sylner(inputs_seq_char, inputs_index_target, params), labels_train
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels)
	cost = tf.reduce_mean(xentropy)
	gradients = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(gradients)

	# Evaluate model
	# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


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
