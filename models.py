#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce

# Import external packages
import numpy as np
import tensorflow as tf

class BaseModel(object):
	"""
	Inherit from this class when implementing new models.
	"""
	def __init__(self):
		pass

	def run_model(self, batch_seq_s, batch_target_idx, **unused_params):
		raise NotImplementedError()

class SylnerModel(BaseModel):
	def __init__(self):
		pass

	def run_model(self, batch_seq_s, batch_target_idx, **unused_params):
		"""
		args:
			batch_seq_s 	: tf.placeholder, shape=[-1, max_length, 3]
			batch_target_idx: tf.placeholder, shape=[-1, 2]
			params 			:
			**unused_params :
		return:
			pred 			: tensor, shape=[-1, number of classes]
		"""
		
		max_length = batch_seq_s,get_shape()[1] # length of the longest sentence in the dataset
		dim_feature = 32

		# Parameters to Learn
		params= {
			'W_conv11': tf.Variable(tf.random_normal([3, 3, 1, 32]), name='W_conv11'),
			'b_conv11': tf.Variable(tf.random_normal([32]), name='W_conv11'),
			# dim_feature = 32
		}


		# Lookup part
		batch_c0_embedded = tf.nn.embedding_lookup(embeddings_cho, batch_seq_s[:,:,0]) # [-1, max_length, 6*2]
		batch_c1_embedded = tf.nn.embedding_lookup(embeddings_jung, batch_seq_s[:,:,1]) # [-1, max_length, 6*2]
		batch_c2_embedded = tf.nn.embedding_lookup(embeddings_jong, batch_seq_s[:,:,2]) # [-1, max_length, 6*2]

		batch_embedded = tf.concat([batch_c0_embedded,
									batch_c1_embedded,
									batch_c2_embedded]) # shape=[-1, max_length, 36]

		batch_conv = tf.reshape(batch_embedded, [-1, 6*6]) # shape=[-1*max_length, 36]
		
		# Conv part
		s = tf.nn.conv2d(batch_conv, params["W_conv11"], strides=[1, 1, 1, 1], padding='SAME')
		s = tf.nn.bias_add(s, params["b_conv11"])
		s = tf.nn.max_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')

		batch_seq_embedded = tf.reshape(s, [-1, max_length, dim_feature]) # shape=[-1, max_length, frame_size]

		# Bi-directional RNN part

		# lstm_cell = tf.contrib.rnn.BasicLSTMCell(size)
		# hidden_state = tf.zeros([batch_size, lstm.state_size])
		# current_state = tf.zeros([batch_size, lstm.state_size])
		# state = hidden_state, current_state
		# output, state = lstm.call(arr_s, state)

		output, state = tf.nn.dynamic_rnn(
			tf.contrib.rnn.GRUCell(num_hidden),
			batch_seq_embedded,
			dtype=tf.float32,
			sequence_length=length(batch_seq_embedded),
		) # shape = [-1, max_length, output_size]

		clipper = tf.zeros(output.get_shape) # shape = [-1, max_length, output_size]
		for i, (idx0, idx1) in enumerate(batch_target_idx):
			row_clipper = tf.zeros([idx0]) + tf.ones([idx1-idx0]) + tf.zeros([max_length])
			clipper = tf.assign(clipper[i], row_clipper)
		
		pred_clipped = tf.multiply(output, clipper) # shape = [-1, max_length, output_size]
		pred = tf.reduce(pred_char_sliced)
		
		# Conditional Random Field
		# To be implemented
		return pred