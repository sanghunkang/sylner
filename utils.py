#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, re, time

# Import external packages
import numpy as np

BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
NONHANGUL_LIST = ["V","X","S","U","L","H","D",".",",","?","!","%","\"","\'","(",")","{","}","[","]","<",">","%","-","·",":","∼"]
# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = NONHANGUL_LIST + ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = NONHANGUL_LIST + ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = NONHANGUL_LIST +  [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def read_data(fpath_data):
	with open(fpath_data, "rb") as fo:
		data = pickle.load(fo)
	return data

def decompose_syllable(syllable):
	"""
	args:
	    syllable    : str, of a single length hangul, which can be decomposed into 3 sub characters
	return:

	"""
	# 유니코드 한글 시작 : 44032, 끝 : 55199
	# BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

	# NONHANGUL_LIST = ["V","X","S","U","L","H","D",".",",","?","!","%","\"","\'","(",")","{","}","[","]","<",">","%","-","·",":","∼"]

	# # 초성 리스트. 00 ~ 18
	# CHOSUNG_LIST = NONHANGUL_LIST + ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

	# # 중성 리스트. 00 ~ 20
	# JUNGSUNG_LIST = NONHANGUL_LIST + ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

	# # 종성 리스트. 00 ~ 27 + 1(1개 없음)
	# JONGSUNG_LIST = NONHANGUL_LIST +  [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

	result = list()
	# 한글 여부 check 후 분리
	if re.match("[ㄱ-ㅣ가-힣]", syllable) is not None:
		char_code = ord(syllable) - BASE_CODE
		char1 = int(char_code / CHOSUNG)
		# result.append(CHOSUNG_LIST[char1 + len(NONHANGUL_LIST)])
		result.append(char1 + len(NONHANGUL_LIST))
		char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
		# result.append(JUNGSUNG_LIST[char2 + len(NONHANGUL_LIST)])
		result.append(char2 + len(NONHANGUL_LIST))
		char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
		# result.append(JONGSUNG_LIST[char3 + len(NONHANGUL_LIST)])
		result.append(char3 + len(NONHANGUL_LIST))
	else:
		try:
			result.append(CHOSUNG_LIST.index(syllable))
			result.append(CHOSUNG_LIST.index(syllable))
			result.append(CHOSUNG_LIST.index(syllable))
		except ValueError:
			result += [0, 0, 0]

	return result

def reformat_data(fpath_data_original):
	with open(fpath_data_original) as fo:
		arr_rec_aux = []
		line = fo.readline()
		while len(line) > 0:
			if line[0] in [";","$"]: arr_rec_aux.append(line)
			line = fo.readline()

		arr_rec= []
		for i in range(0, len(arr_rec_aux), 2):
			x1 = arr_rec_aux[i][2:].strip()
			x2 = arr_rec_aux[i+1][1:].strip()
			if "<" not in x1: arr_rec.append([x1,x2])
	return arr_rec

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

def write_data_ready(arr_rec_a):
	str_fwrite = ""
	arr_len_seq = []
	count = 0
	arr_recrec = []
	
	for rec in arr_rec_a:
		sen_raw, sen_labelled = rec[0], rec[1]
		sen_raw_preserved = sen_raw
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
			arr_len_seq.append(len(sen_raw))
			arr_recrec.append(sen_raw_preserved)		
			count += 1
	shape_data = (count, 341)
	return shape_data, str_fwrite, arr_recrec

def digitize_data(arr_rec_str, shape_data):
	arr_label = ['TI', 'OG', 'PS', 'LC', 'DT']
	
	data = np.zeros(shape=(shape_data[0], shape_data[1]*(3+1) + 5), dtype=np.int32)
	for i, rec in enumerate(arr_rec_str):
		# print(rec)
		sen_raw = rec.split(";")[0]

		# Input row
		seq_digitized = []
		for x in sen_raw: seq_digitized += decompose_syllable(x)
		seq_digitized = np.asarray(seq_digitized, dtype=np.int32)
		zero_filler = np.zeros(shape=(shape_data[1]*3 - seq_digitized.shape[0]))
		X = np.hstack([seq_digitized, zero_filler])

		# Clipper
		zeros0 = np.zeros(shape=int(rec.split(";")[1]))
		ones = np.ones(shape=int(rec.split(";")[2]) - int(rec.split(";")[1]))
		zeros1 = np.zeros(shape=shape_data[1]- int(rec.split(";")[2]))
		clipper = np.hstack([zeros0, ones, zeros1])
		
		# Label
		label_onehot = [0 for i in range(len(arr_label))]
		label_onehot[arr_label.index(rec.split(";")[3].strip())] = 1
		label_onehot = np.asarray(label_onehot, dtype=np.int32)

		record = np.hstack([X, clipper, label_onehot])
		data[i] = record
	return data

def nikl_to_nparray(fpath):
	arr_rec = reformat_data(fpath)
	shape_data, str_fwrite, arr_recrec = write_data_ready(arr_rec)

	arr_rec_str = str_fwrite.split("\n")[:-1]
	data = digitize_data(arr_rec_str, shape_data)
	# return data
	return data, arr_rec_str, arr_recrec

def write_pickle(data, fpath):
	with open(fpath, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# data_train, _  = nikl_to_nparray("../../dev-data/sylner/2016klpNER.base_train")
# print(data_train.shape)