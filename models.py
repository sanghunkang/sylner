#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce

# Import external packages
import numpy as np
import tensorflow as tf

import hangulvars

BASE_CODE = hangulvars.BASE_CODE
CHOSUNG = hangulvars.CHOSUNG
JUNGSUNG = hangulvars.JUNGSUNG
NONHANGUL_LIST = hangulvars.NONHANGUL_LIST
CHOSUNG_LIST = hangulvars.CHOSUNG_LIST
JUNGSUNG_LIST = hangulvars.JUNGSUNG_LIST
JONGSUNG_LIST = hangulvars.JONGSUNG_LIST

class BaseModel(object):
	"""
	Inherit from this class when implementing new models.
	"""
	def __init__(self):
		pass

	def run_model(self, batch_seq_s, batch_clipper, **unused_params):
		raise NotImplementedError()

class SylnerModel(BaseModel):
	def __init__(self):
		pass

	def run_model(self, batch_seq_s, batch_clipper, **unused_params):
		"""
		args:
			batch_seq_s 	: tf.placeholder, shape=[-1, max_length, 3]
			batch_clipper 	: tf.placeholder, shape=[-1, max_length, num_class]
			params 			:
			**unused_params :
		return:
			pred 			: tensor, shape=[-1, num_class]
		"""
		batch_size = batch_seq_s.get_shape()[0]
		max_length = int(batch_seq_s.get_shape()[1]) # length of the longest sentence in the dataset
		print(batch_size, max_length)
		

		# Parameters to Learn
		dim_embed = 12
		embedding_cho = tf.Variable(tf.random_uniform([len(CHOSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_cho")
		embedding_jung = tf.Variable(tf.random_uniform([len(JUNGSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_jung")
		embedding_jong = tf.Variable(tf.random_uniform([len(JONGSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_jong")

		params= {
			'W_conv11': tf.Variable(tf.random_normal([3, 3, 1, 32]), name='W_conv11'),
			'b_conv11': tf.Variable(tf.random_normal([32]), name='W_conv11'),
			# dim_feature = 32
		}


		# Lookup part
		batch_c0_embedded = tf.nn.embedding_lookup(embedding_cho, batch_seq_s[:,:,0]) # [-1, max_length, 6*2]
		batch_c1_embedded = tf.nn.embedding_lookup(embedding_jung, batch_seq_s[:,:,1]) # [-1, max_length, 6*2]
		batch_c2_embedded = tf.nn.embedding_lookup(embedding_jong, batch_seq_s[:,:,2]) # [-1, max_length, 6*2]

		batch_embedded = tf.concat([batch_c0_embedded,
									batch_c1_embedded,
									batch_c2_embedded], axis=2) # shape=[-1, max_length, 36]

		batch_conv = tf.reshape(batch_embedded, [-1, 6*6]) # shape=[-1*max_length, 36]
		batch_conv = tf.reshape(batch_conv, [-1, 6, 6, 1])
		
		# Conv part
		s = tf.nn.conv2d(batch_conv, params["W_conv11"], strides=[1, 1, 1, 1], padding='SAME')
		s = tf.nn.bias_add(s, params["b_conv11"])
		s = tf.nn.max_pool(s, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')

		dim_feature = 32
		batch_seq_embedded = tf.reshape(s, [-1, max_length, dim_feature]) # shape=[-1, max_length, frame_size]

		# Bi-directional RNN part
		with tf.variable_scope("GRULayer1"):
			cell_fw = tf.nn.rnn_cell.GRUCell(20)
			# output1, state1 = tf.nn.dynamic_rnn(cell_fw, batch_seq_embedded, dtype=tf.float32)
		
		with tf.variable_scope("GRULayer2"):
			cell_bw = tf.nn.rnn_cell.GRUCell(20)
			# output, state2 = tf.nn.dynamic_rnn(cell_bw, output1, dtype=tf.float32)

		with tf.variable_scope("GRULayer3"):
			cell_end = tf.nn.rnn_cell.GRUCell(5)

		output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
														cell_bw,
														batch_seq_embedded,
														dtype=tf.float32,
														scope=None)
		print(output[0].shape, output[1].shape)
		output_concat = tf.concat(output, axis=2)
		print(output_concat.shape)

		output, state2 = tf.nn.dynamic_rnn(cell_end, output_concat, dtype=tf.float32)

		# W_proj = tf.Variable(tf.random_normal([20*2, 5]), name='W_proj')

		# output = tf.matmul(output_concat, W_proj*int(output_concat.shape[1]))

		batch_clipper_expanded = tf.stack([batch_clipper]*5, axis=2)
		pred_clipped = tf.multiply(output, batch_clipper_expanded) # shape = [-1, max_length, output_size]
		pred = tf.reduce_sum(pred_clipped, axis=1)
		print(pred)
		
		# Conditional Random Field
		# To be implemented
		return pred