[한국어 README](./README.ko.md)
# SylNER: Syllable-based Named Entity Recognition for Korean Language
[sylner](./sylner.png)
Submission to 2017 Korean Language Expo

#### Acknowledgement
(https://github.com/neotune/python-korean-handler)
Here, I could borrow a handy function to decompose syllables into characters.
([Kim et al. 2015](somelink))
This is where our most inspiration is from.

#### Dependencies
* Python 3.5.3 
* TensorFlow 1.3 or higher
* Numpy 1.13.1 or higher
* Matplolib 2.0.2 or higher

#### Description
More solid(?) article-style explanation can be found at ()(English) and ()(Korean)
Korean language is a morphologically rich length, to the extent, that in case of "hyeongyongsa" - which correspond to combination of a be-verb and an adjective in English language - and "dongsa" - which corresponds to verb in English language.  But combination of "josa"s - which in most cases act as an prepositiom... The task our model is doing is called "Named Entity Recognition". So the part "NER" of our model was undisputable. 

#### Usage
There's a copyright issue regarding the data  . The dataset is provided by ... if you with to recall the test, you may contact (2017klexpo@...com)

Real-time implementation of NLP requires less computing  . It is an issue espe  .. Since most of the so called POS(Part of Speach) taggers rely on  . But the latest implementations of computerised language processing utilise GPU, which run best when external pacakges are excluded in implementations as much as possibile. It is generally regarded as a slowing factor to use high-level wrappers in implementations of deep learning algo

This technical environment demand us a task of escaping from "morpheme first -> that do whatever" protocol in Korean natural language processing. The main contribution of this research was intended to handle this issue. The research is inspired by [Kim et al. 2015](somelink), which gave us some hint to completely exclude usage of POS taggers and morpheme taggers, plus the representation format of Korean language called "Hangul".

#### Usage
###### Ubuntu (not tested, but presumably in OS X as well)
```
python3 train.py --fpath_data_train=/path/to/trainset --fpath_data_eval=/path/to/trainset --logdir=/path/to/logs
```
Of course you may confer ```-f```flags for details
training p-
The organiser of the contest doesn't seem to want to make public of the data it provided for the contest. For more information, please contact 

#### Some Fun
For instance, my name looks like this.

To keep the integrity of our of research, we rendered the provided dataset  to fit..

Chinese characters are converted into Korean letters with corresponding pronunciations to
Roman alphabets ...?
Numbers are converted into Korean letters with corresponding pronunciations to Chinese style of reading such numbers
Quotes, commas, periods, spaces, are repla ... That whenever... Basically in any sound-representing languages, every token has some effect on physical pronunciation. For instance, spaces usually indicate a point to brethee, the existance of comma may result in extra pause between spaces, quotes   .   Depending on the specific settings of ... we implemeted each mark slightly differently, but the baseline was to associate each symbol to a single syllable.
Some symbols are assumed to be the same. For instance, quotes and double quotes, parentheses and brackets.


* Each position (consonant1, vowel, consonant2) is only a channel of a syllable . To put it into an implementation, we used 1d convolution of 3 channels at input stage. 
* Each potision is somehow related to sounds of other positions. To  ... we used 2d convolution on feeding inputs into networks.

####
Chinese characters are symbolic. Each letter on its own contains some meaning. However, the size of the vocabulary is huge.  We can partly appropriate the advantage of Chinese writing system but succesfully avoid the problem of estimating a huge dimension matrix by constructing a syllable with three subsounds.


####
* Input: Sentence/Position(Index + length)
* Output: Probability that a word belongs to the category



