#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, time

# Import external packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import SylnerModel
import utils, hangulvars 

BASE_CODE = utils.BASE_CODE
CHOSUNG = utils.CHOSUNG
JUNGSUNG = utils.JUNGSUNG
NONHANGUL_LIST = utils.NONHANGUL_LIST
CHOSUNG_LIST = utils.CHOSUNG_LIST
JUNGSUNG_LIST = utils.JUNGSUNG_LIST
JONGSUNG_LIST = utils.JONGSUNG_LIST

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("string", "실너", "The directory to save the model files in.")
tf.flags.DEFINE_string("logdir", "./logs_gru32_full", "The directory to save the model files in.")

# Deactivate minor warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# BUILDING THE COMPUTATIONAL GRAPH
graph = tf.Graph()
with graph.as_default():
	dim_embed = 12

	embedding_cho = tf.Variable(tf.random_uniform([len(CHOSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_cho", dtype=tf.float32)
	embedding_jung = tf.Variable(tf.random_uniform([len(JUNGSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_jung")
	embedding_jong = tf.Variable(tf.random_uniform([len(JONGSUNG_LIST), dim_embed], -1.0, 1.0), name="embeddings_jong")

	saver = tf.train.Saver()

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):
	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(graph=graph, config=config) as sess:
		tf.global_variables_initializer().run()

		# Load Checkpoint Data
		saver.restore(sess, './{0}/checkpoint.ckpt'.format(FLAGS.logdir))

		sen = FLAGS.string
		a = 100 + 10*len(sen)
		for i, s in enumerate(sen):
			arr_c = utils.decompose_syllable(s)
			
			c1 = embedding_cho.eval()[arr_c[0]]
			c2 = embedding_jung.eval()[arr_c[1]]
			c3 = embedding_jong.eval()[arr_c[2]]
			
			c1 = np.reshape(c1, [2, 6])
			c2 = np.reshape(c2, [2, 6])
			c3 = np.reshape(c3, [2, 6])

			s_embedded = np.vstack([c1, c2, c3])

			plt.subplot(a +i+1)
			plt.imshow(s_embedded, norm=matplotlib.colors.NoNorm(), cmap="gray")
			plt.axis('off')

		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	tf.app.run()