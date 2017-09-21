#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle, re

import numpy as np

import hangulvars, utils

BASE_CODE = hangulvars.BASE_CODE
CHOSUNG = hangulvars.CHOSUNG
JUNGSUNG = hangulvars.JUNGSUNG
NONHANGUL_LIST = hangulvars.NONHANGUL_LIST
CHOSUNG_LIST = hangulvars.CHOSUNG_LIST
JUNGSUNG_LIST = hangulvars.JUNGSUNG_LIST
JONGSUNG_LIST = hangulvars.JONGSUNG_LIST

def generate_arr_index_target(sen_labelled):
	arr_index_target = []
	padding = 0
	idx_searched = 0
	while True:
		try:
			idx0 = sen_labelled.index("<", idx_searched)
			idx1 = sen_labelled.index(">", idx_searched)
			label = sen_labelled[idx0:idx1].split(":")[1]
			
			arr_index_target.append([idx0 - padding, idx1 - padding - 2 - len(label), label])
			
			idx_searched = idx1 + 1
			padding += 3 + len(label)
		except ValueError:
			break

	return arr_index_target

def reformat_data(fpath_data_original):
	with open(fpath_data_original) as fo:
		arr_rec = []
		line = fo.readline()
		while len(line) > 0:
			if line[0] in [";","$"]: arr_rec.append(line)
			line = fo.readline()

		# print(range(len(arr_rec), 2))
		arr_rec_a = []
		for i in range(0, len(arr_rec), 2):
			x1 = arr_rec[i][2:].strip()
			x2 = arr_rec[i+1][1:].strip()
			if "<" not in x1: arr_rec_a.append([x1,x2])

		return arr_rec_a
		# return (count, max(arr_len_seq))

def write_data_ready(arr_rec_a, fpath_data_ready):
	str_fwrite = ""
	arr_len_seq = []
	count = 0
	
	for rec in arr_rec_a:
		sen_raw, sen_labelled = rec[0], rec[1]
		# print(sen_raw, sen_labelled)
		sen_raw = re.sub(r"([A-Z])", "U", sen_raw) # Uppercase letters
		sen_raw = re.sub(r"([a-z])", "L", sen_raw) # Lowercase letters
		sen_raw = re.sub(r"([\u4e00-\u9fff])", "H", sen_raw) # Hanja
		sen_raw = re.sub(r"([0-9])", "D", sen_raw) # Digits
		sen_raw = re.sub(r" ", "S", sen_raw) # Whitespace
		# sen_raw = "B" + sen_raw + "E"
		# print(sen_raw)

		arr_index_target = generate_arr_index_target(sen_labelled)
		for index_target in arr_index_target:
			str_fwrite = "{0}{1};{2};{3};{4}\n".format(str_fwrite, sen_raw, index_target[0], index_target[1], index_target[2])
			# print(sen_raw[index_target[0]:index_target[1]], index_target[2])
			arr_len_seq.append(len(sen_raw))
		
			count += 1
		# line = fo.readline()
	shape_data = (count, max(arr_len_seq))
	return shape_data, str_fwrite


def digitize_data(fpath, shape_data):
	arr_label = ['TI', 'OG', 'PS', 'LC', 'DT']
	with open(fpath, "r", encoding="utf-8") as fo:		
		data = np.zeros(shape=(shape_data[0], shape_data[1]*(3+1) + 5), dtype=np.int32)
		for i in range(shape_data[0]):
			line = fo.readline()
			sen_raw = line.split(";")[0]

			# print(sen_raw)

			# Input row
			seq_digitized = []
			for x in sen_raw: seq_digitized += utils.decompose_syllable(x)
			seq_digitized = np.asarray(seq_digitized, dtype=np.int32)
			zero_filler = np.zeros(shape=(shape_data[1]*3 - seq_digitized.shape[0]))
			X = np.hstack([seq_digitized, zero_filler])

			# Clipper
			zeros0 = np.zeros(shape=int(line.split(";")[1]))
			ones = np.ones(shape=int(line.split(";")[2]) - int(line.split(";")[1]))
			zeros1 = np.zeros(shape=shape_data[1]- int(line.split(";")[2]))
			clipper = np.hstack([zeros0, ones, zeros1])
			
			# Label
			label_onehot = [0 for i in range(len(arr_label))]
			label_onehot[arr_label.index(line.split(";")[3].strip())] = 1
			label_onehot = np.asarray(label_onehot, dtype=np.int32)

			record = np.hstack([X, clipper, label_onehot])
			data[i] = record
	return data

fpath_data_original = "../../dev-data/sylner/2016klpNER.base_train"
fpath_data_raw = "../../dev-data/sylner/base_train_modified.csv"
fpath_data_ready = "../../dev-data/sylner/base_train_ready.csv"
fpath_pickle = "../../dev-data/sylner/base_train.pickle"

arr_rec = reformat_data(fpath_data_original)
shape_data, str_fwrite = write_data_ready(arr_rec, fpath_data_ready)

with open(fpath_data_ready, "w", encoding="utf-8") as fo:
	fo.write(str_fwrite)

data = digitize_data(fpath_data_ready, shape_data)
print(data.shape)
utils.write_pickle(data, fpath_pickle)