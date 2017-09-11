#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
from functools import reduce

# Import external packages
import numpy as np
import tensorflow as tf

class BaseModel(object):
	"""Inherit from this class when implementing new models."""

	def create_model(self, inputs_seq_char, inputs_index_target, params, **unused_params):
		raise NotImplementedError()

class TargetSubtractionModel(BaseModel):
	def __init__(self):
		pass

	def model_sylner(self, inputs_seq_char, inputs_index_target, params, **unused_params):
		X = inputs_seq_char
		index0, index1 = inputs_index_target
		# Convert the raw input sequence into sequence of 3 channel embeddings
		X1_including_target = tf.nn.embedding_lookup(embeddings_cho, X[-1, 0, :])
		X2_including_target = tf.nn.embedding_lookup(embeddings_jung, X[-1, 1, :])
		X3_including_target = tf.nn.embedding_lookup(embeddings_jong, X[-1, 2, :])

		tf.zeros(shape, dtype=tf.float32)

		X1_excluding_target = X1_including_target
		X2_excluding_target = 
		X3_excluding_target = 
		# Do the same for target words
		# X1_target = tf.nn.embedding_lookup(embeddings_cho, X[-1, 0, index0:index1])
		# X2_target = tf.nn.embedding_lookup(embeddings_jung, X[-1, 1, index0:index1])
		# X3_target = tf.nn.embedding_lookup(embeddings_jong, X[-1, 2, index0:index1])

		X_context = tf.concatenate([X1_context, X2_context, X3_context])
		X_target = tf.concatenate([X1_target, X2_target, X3_target])

		# Convnets before entering 
		x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)

		words_in_dataset = tf.placeholder(tf.float32, [num_batches, batch_size, num_features])
		
		# Feature generation including target word
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(size)
		hidden_state = tf.zeros([batch_size, lstm.state_size])
		current_state = tf.zeros([batch_size, lstm.state_size])
		state = hidden_state, current_state

		for current_batch_of_words in words_in_dataset:
			# The value of state is updated after processing each batch of words.
			output, state = lstm.call(current_batch_of_words, state)

			# The LSTM output can be used to make next word predictions
			# logits = tf.matmul(output, softmax_w) + softmax_b
			# probabilities.append(tf.nn.softmax(logits))
			# loss += loss_function(probabilities, target_words)

		# Feature generation excluding target word
		for current_batch_of_words in words_in_dataset:
			# The value of state is updated after processing each batch of words.
			output, state = lstm.call(current_batch_of_words, state)

		vec_feature_including_target = 
		vec_feature_excluding_target = 
		vec_feature_diff = tf.subtract(vec_feature_including_target, vec_feature_excluding_target, name="subtraction")

		fc = tf.add(tf.matmul(vec_feature_diff, params["fc1_W"]), params["fc1_b"])
		fc = tf.contrib.layers.batch_norm(fc)
		fc = tf.nn.relu(fc)

		fc = tf.add(tf.matmul(vec_feature_diff, params["fc2_W"]), params["fc2_b"])
		fc = tf.contrib.layers.batch_norm(fc)
		fc = tf.nn.relu(fc)
		pred = fc # a tensor of shape [-1, 5]
		return pred
	
	

class TargetAppendedModel(BaseModel):
	def __init__(self):
		pass

	def model_sylner(self, inputs_seq_char, inputs_index_target, params, **unused_params):
		X = inputs_seq_char
		index0, index1 = inputs_index_target
		# Convert the raw input sequence into sequence of 3 channel embeddings
		X1_context = tf.nn.embedding_lookup(embeddings_cho, X[-1, 0, :])
		X2_context = tf.nn.embedding_lookup(embeddings_jung, X[-1, 1, :])
		X3_context = tf.nn.embedding_lookup(embeddings_jong, X[-1, 2, :])

		# Do the same for target words
		# X1_target = tf.nn.embedding_lookup(embeddings_cho, X[-1, 0, index0:index1])
		# X2_target = tf.nn.embedding_lookup(embeddings_jung, X[-1, 1, index0:index1])
		# X3_target = tf.nn.embedding_lookup(embeddings_jong, X[-1, 2, index0:index1])