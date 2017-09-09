#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
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

    # 초성 리스트. 00 ~ 18
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    result = list()
    keyword = syllable
    # for keyword in split_keyword_list:
    # 한글 여부 check 후 분리
    if re.match("[ㄱ-ㅣ가-힣]", keyword) is not None:
        char_code = ord(keyword) - BASE_CODE
        char1 = int(char_code / CHOSUNG)
        result.append(CHOSUNG_LIST[char1])
        char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
        result.append(JUNGSUNG_LIST[char2])
        char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
        result.append(JONGSUNG_LIST[char3])
    else:
        result.append(keyword)
    return result

with open("../../dev-data/2017klexpo/2016klpNER.base_train") as fo:
	aa = fo.read()
	bb = aa.split(";")
	for i in bb[1:2]:
		bbb=i.split("\n")
		print(len(bbb[0][1]))
		print(bbb[1])
		print(len(bbb))
		for x in bbb[0]:			
			result = decompose_syllable(x)
			print("".join(result),)
		print("\n")
	print(len(bb))