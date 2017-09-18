#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

import numpy as np
"""
    초성 중성 종성 분리 하기
	유니코드 한글은 0xAC00 으로부터
	초성 19개, 중상21개, 종성28개로 이루어지고
	이들을 조합한 11,172개의 문자를 갖는다.
	한글코드의 값 = ((초성 * 21) + 중성) * 28 + 종성 + 0xAC00
	(0xAC00은 'ㄱ'의 코드값)
	따라서 다음과 같은 계산 식이 구해진다.
	유니코드 한글 문자 코드 값이 X일 때,
	초성 = ((X - 0xAC00) / 28) / 21
	중성 = ((X - 0xAC00) / 28) % 21
	종성 = (X - 0xAC00) % 28
	이 때 초성, 중성, 종성의 값은 각 소리 글자의 코드값이 아니라
	이들이 각각 몇 번째 문자인가를 나타내기 때문에 다음과 같이 다시 처리한다.
	초성문자코드 = 초성 + 0x1100 //('ㄱ')
	중성문자코드 = 중성 + 0x1161 // ('ㅏ')
	종성문자코드 = 종성 + 0x11A8 - 1 // (종성이 없는 경우가 있으므로 1을 뺌)
"""


def decompose_syllable(syllable):
	"""
	args:
	    syllable    : str, of a single length hangul, which can be decomposed into 3 sub characters
	return:

	"""
	# 유니코드 한글 시작 : 44032, 끝 : 55199
	BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

	NON_HANGUL = ["E","X","S","U","L","H","D",".",",","?","!","%","\"","\'","(",")","{","}","[","]","<",">","%","-","·",":","∼"]

	# 초성 리스트. 00 ~ 18
	CHOSUNG_LIST = NON_HANGUL + ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

	# 중성 리스트. 00 ~ 20
	JUNGSUNG_LIST = NON_HANGUL + ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

	# 종성 리스트. 00 ~ 27 + 1(1개 없음)
	JONGSUNG_LIST = NON_HANGUL +  [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

	result = list()
	# 한글 여부 check 후 분리
	if re.match("[ㄱ-ㅣ가-힣]", syllable) is not None:
		char_code = ord(syllable) - BASE_CODE
		char1 = int(char_code / CHOSUNG)
		# result.append(CHOSUNG_LIST[char1 + len(NON_HANGUL)])
		result.append(char1 + len(NON_HANGUL))
		char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
		# result.append(JUNGSUNG_LIST[char2 + len(NON_HANGUL)])
		result.append(char2 + len(NON_HANGUL))
		char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
		# result.append(JONGSUNG_LIST[char3 + len(NON_HANGUL)])
		result.append(char3 + len(NON_HANGUL))
	else:
		try:
			result.append(CHOSUNG_LIST.index(syllable))
			result.append(CHOSUNG_LIST.index(syllable))
			result.append(CHOSUNG_LIST.index(syllable))
		except ValueError:
			result += [0, 0, 0]

	return result

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

def write_data_ready(fpath_data_raw, fpath_data_ready):
	with open(fpath_data_raw, "r") as fo:
		str_fwrite = ""
		line = fo.readline()
		arr_len_seq = []
		count = 0
		while len(line) > 0:
			sen_raw = line.split(";")[0]
			sen_labelled = line.split(";")[1]
			
			sen_raw = re.sub(r"([A-Z])", "U", sen_raw) # Uppercase letters
			sen_raw = re.sub(r"([a-z])", "L", sen_raw) # Lowercase letters
			sen_raw = re.sub(r"([\u4e00-\u9fff])", "H", sen_raw) # Hanja
			sen_raw = re.sub(r"([0-9])", "D", sen_raw) # Digits
			sen_raw = re.sub(r" ", "S", sen_raw) # Whitespace

			arr_index_target = generate_arr_index_target(sen_labelled)
			for index_target in arr_index_target:
				str_fwrite = "{0}{1};{2};{3};{4}\n".format(str_fwrite, sen_raw, index_target[0], index_target[1], index_target[2])
				# print(sen_raw[index_target[0]:index_target[1]], index_target[2])
				arr_len_seq.append(len(sen_raw))
			
			count += 1
			line = fo.readline()

		return (count, max(arr_len_seq)*3 + 2 + 5)

	with open(fpath_data_ready, "w", encoding="utf-8") as fo:
		fo.write(str_fwrite)

def digitize_data(fpath, shape_data):
	arr_label = ['TI', 'OG', 'PS', 'LC', 'DT']
	with open(fpath, "r", encoding="utf-8") as fo:
		line = fo.readline()
		
		data = np.zeros(shape=shape_data, dtype=np.int32)
		while len(line) > 0:
			# Variable length data
			sen_raw = line.split(";")[0]

			print(sen_raw)
			seq_digitized = []
			for x in sen_raw: seq_digitized += decompose_syllable(x)

			seq_digitized = np.asarray(seq_digitized, dtype=np.int32)
			zero_filler = np.zeros(shape=(shape_data[1] - seq_digitized.shape[0]))
			

			# Fixed length data
			idx_target = np.asarray([line.split(";")[1], line.split(";")[2]], dtype=np.int32)
			
			label_onehot = [0 for i in range(len(arr_label))]
			label_onehot[arr_label.index(line.split(";")[3].strip())] = 1
			label_onehot = np.asarray(label_onehot, dtype=np.int32)

			print(seq_digitized, idx_target, label_onehot)
			record = np.hstack([seq_digitized, zero_filler, idx_target, label_onehot])
			print(record)
			line = fo.readline()
	return data

fpath_data_raw = "../../dev-data/sylner/base_train_modified.csv"
fpath_data_ready = "../../dev-data/sylner/base_train_ready.csv"

shape_data = write_data_ready(fpath_data_raw, fpath_data_ready)
digitize_data(fpath_data_ready, shape_data)