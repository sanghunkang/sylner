# sylner
(Potential) Submission to 2017KLExpo

#### Acknowledgement
(https://github.com/neotune/python-korean-handler)

#### Description
Korean language is a morphologically rich length, to the extent, that in case of "hyeongyongsa" - which correspond to combination of a be-verb and an adjective in English language - and "dongsa" - which corresponds to verb in English language.  But combination of "josa"s - which in most cases act as an prepositiom...

Real-time implementation of NLP requires less computing  . It is an issue espe  .. Since most of the so called POS(Part of Speach) taggers rely on  . But the latest implementations of computerised language processing utilise GPU, which run best when external pacakges are excluded in implementations as much as possibile. It is generally regarded as a slowing factor to use high-level wrappers in implementations of deep learning algo

This technical environment demand us a task of escaping from "morpheme first -> that do whatever" protocol in Korean natural language processing. The main contribution of this research was intended to handle this issue. The research is inspired by [Kim et al. 2015](somelink), which gave us some hint to completely exclude usage of POS taggers and morpheme taggers, plus the representation format of Korean language called "Hangul".


To keep the integrity of our of research, we rendered the provided dataset  to fit..

Chinese characters are converted into Korean letters with corresponding pronunciations to
Roman alphabets ...?
Numbers are converted into Korean letters with corresponding pronunciations to Chinese style of reading such numbers
Quotes, commas, periods, spaces, are repla ... That whenever... Basically in any sound-representing languages, every token has some effect on physical pronunciation. For instance, spaces usually indicate a point to brethee, the existance of comma may result in extra pause between spaces, quotes   .   Depending on the specific settings of ... we implemeted each mark slightly differently, but the baseline was to associate each symbol to a single syllable.
Some symbols are assumed to be the same. For instance, quotes and double quotes, parentheses and brackets.


* Each position (consonant1, vowel, consonant2) is only a channel of a syllable . To put it into an implementation, we used 1d convolution of 3 channels at input stage. 
* Each potision is somehow related to sounds of other positions. To  ... we used 2d convolution on feeding inputs into networks.


####
Input: Sentence/Position(Index + length)
Output: Probability that a word belongs to the category



