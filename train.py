#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, time

# Import external packages
import numpy as np
import tensorflow as tf

from models import SylnerModel
import hangulvars 
import utils

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
tf.flags.DEFINE_integer("batch_size", 64, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_class", 5, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate, epsilon")

def feed_data(data, batch_size, num_class):
	np.random.shuffle(data)
	batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	len_seq = (data.shape[1] - num_class)//4

	return {batch_seq_s: np.reshape(batch[:, :len_seq*3],[-1,len_seq,3]),
			batch_clipper: batch[:, len_seq*3:len_seq*4],
			batch_labels: batch[:, len_seq*4:]}

# BUILDING THE COMPUTATIONAL GRAPH
graph = tf.Graph()
with graph.as_default():
	sylnerModel = SylnerModel()

	# Hyperparameters
	learning_rate = FLAGS.learning_rate
	display_step = 10

	# tf Graph input
	num_class = FLAGS.num_class # Normal or Abnormal

	# Placeholders
	count_step = tf.Variable(0, name="count_step")
	
	FLAGS.batch_size
	max_length = 341 # Temporary hard coding

	batch_seq_s = tf.placeholder(tf.int32, shape=[None, max_length, 3])
	batch_clipper = tf.placeholder(tf.float32, shape=[None, max_length])
	batch_labels = tf.placeholder(tf.int32, shape=[None, num_class]) # One-hot encoding
	
	# Define loss, compute gradients
	pred = sylnerModel.run_model(batch_seq_s, batch_clipper)
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
	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	data = utils.read_data("../../dev-data/sylner/base_train.pickle")
	print(data.shape)
	data_train = data[:5000]
	data_eval = data[5000:]

	# Run session
	with tf.Session(graph=graph, config=config) as sess:
		tf.global_variables_initializer().run()
		train_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "train"), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "test"), sess.graph)

		num_steps = FLAGS.num_steps
		batch_size = FLAGS.batch_size
		num_class = FLAGS.num_class

		# Load Checkpoint Data
		try:
			saver.restore(sess, './{0}/checkpoint.ckpt'.format(FLAGS.logdir))
			print('Model restored')
			epoch_saved = count_step.eval()
		except tf.errors.NotFoundError:
			print('No saved model found')
			epoch_saved = 0
		except tf.errors.InvalidArgumentError:
			print('Model structure has been changed. Rebuild model')
			epoch_saved = 0

		# Train Loop
		for epoch in range(epoch_saved, epoch_saved + num_steps):
			# Run optimization op (backprop)
			summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_data(data_train, batch_size, num_class))
			train_writer.add_summary(summary, epoch)

			summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_data(data_eval, batch_size, num_class))
			test_writer.add_summary(summary, epoch)
			print("Epoch {0}: Loss= {1:.6f}, Train Accuracy= {2:.5f}, Validation Accuracy= {3:.5f}".format(epoch, loss_train, acc_train, acc_test))

		print("Optimisation Finished!")

		# Save Checkpoint Data
		epoch_new = epoch_saved + num_steps
		sess.run(count_step.assign(epoch_new))
		fpath_ckpt = saver.save(sess, "./{0}/checkpoint.ckpt".format(FLAGS.logdir))
		print("Model saved in file: {0}".format(fpath_ckpt))

if __name__ == "__main__":
	tf.app.run()
