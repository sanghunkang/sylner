#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce
import pickle, time

# Import external packages
import numpy as np
import tensorflow as tf

from models import SylnerModel
import hangulvars 

BASE_CODE = hangulvars.BASE_CODE
CHOSUNG = hangulvars.CHOSUNG
JUNGSUNG = hangulvars.JUNGSUNG
NONHANGUL_LIST = hangulvars.NONHANGUL_LIST
CHOSUNG_LIST = hangulvars.CHOSUNG_LIST
JUNGSUNG_LIST = hangulvars.JUNGSUNG_LIST
JONGSUNG_LIST = hangulvars.JONGSUNG_LIST

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("fpath_data_train", "../../dev-data/sylner/base_train.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_string("fpath_data_eval", "../../dev-datasylner/base_eval1.pickle", "The directory to save the model files in.")
tf.flags.DEFINE_integer("batch_size", 128, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_class", 5, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_integer("learning_rate", 0.001, "Learning rate, epsilon")


def read_data(fpath_data):
	with open(fpath_data, "rb") as fo:
		data = pickle.load(fo)
		np.random.shuffle(data)
	return data

def feed_data(data, batch_size, num_class):
	# batch = np.zeros(shape=(batch_size, len_input+len(stack_data)), dtype=np.float32)
	batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	len_seq = (data.shape[1] - num_class)//4
	print(len_seq)
	return {batch_seq_s: np.reshape(batch[:, :len_seq*3],[-1,len_seq,3]),
			batch_clipper: batch[:, len_seq*3:len_seq*4],
			batch_labels: batch[:, len_seq*4:]}

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
	
	FLAGS.batch_size
	max_length = 217 # Temporary hard coding

	batch_seq_s = tf.placeholder(tf.int32, shape=[None, max_length, 3])
	batch_clipper = tf.placeholder(tf.float32, shape=[None, max_length])
	# batch_target_idx = tf.placeholder(tf.int32, shape=[None, 2]) # Start-index and end-index
	batch_labels = tf.placeholder(tf.int32, shape=[None, num_class]) # One-hot encoding
	
	# Define loss, compute gradients
	pred = sylnerModel.run_model(batch_seq_s, batch_clipper)
	print(pred.get_shape())
	print(batch_labels.get_shape())
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=batch_labels)
	cost = tf.reduce_mean(xentropy)
	gradients = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(gradients)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(batch_labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# Write summary for Tensorboard
	tf.summary.scalar('accuracy', accuracy)
	merged = tf.summary.merge_all()

	# Define saver to save/restore trained result
	saver = tf.train.Saver()

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):

	# Define saver
	# saver = tf.train.Saver()

	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	data = read_data("../../dev-data/sylner/base_train.pickle")
	# Run session
	with tf.Session(graph=graph, config=config) as sess:
		tf.global_variables_initializer().run()
		summaries_dir = './logs'
		train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)

		num_steps = FLAGS.num_steps
		batch_size = FLAGS.batch_size
		num_class = FLAGS.num_class

		epoch_saved = 0
		for epoch in range(epoch_saved, epoch_saved + num_steps):
			# Run optimization op (backprop)
			summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_data(data, batch_size, num_class))
			train_writer.add_summary(summary, epoch)

			# summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_data(data, batch_size, num_class))
			# test_writer.add_summary(summary, epoch)
			# print("Accuracy at step {0}: {1}".format(epoch, acc_test))

			# if epoch % display_step == 0:
			print("Epoch {0}, Minibatch Loss= {1:.6f}, Train Accuracy= {2:.5f}".format(epoch, loss_train, acc_train))

		print("Optimisation Finished!")

if __name__ == "__main__":
	tf.app.run()
