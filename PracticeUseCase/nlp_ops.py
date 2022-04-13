import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from numba import jit


def remove_features(word, tagger, lmtzr):
    """Returns a word after it has been checked to see if it is worth keeping"""
    function_list = [remove_stop_words, remove_puncuation, remove_numbers, filter_tag_pos, lemmatize_word,
                     remove_short_words]
    # lowercase
    word = word.lower()
    # iterate through functions and stop if the word gets thrown out
    #htd
    for func in function_list:
        if func == filter_tag_pos:
            word, tagged_text = func(word, tagger)
            print("Stopped words :", word)
            print("\n")
            print("tagged_text :", tagged_text)
        elif func == lemmatize_word:
            word = func(tagged_text, lmtzr)
            print("lemmatize_word :", word)

        else:
            word = func(word)
            print("{} :".format(func), word)
        if word.isspace() or word == '':
            break
    return word


@jit
def filter_tag_pos(word, tagger):
    """Tag Part of Speach keep only verbs, nouns and adjectives"""
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags
    tagged_text = tagger.tag([word])
    # word & tag tuple
    if tagged_text[0][1] not in nltk_tags:
        word = ''
    return word, tagged_text


@jit
def lemmatize_word(tagged_text, lmtzr):
    if tagged_text[0][1][0].lower() == 'v':
        word = lmtzr.lemmatize(tagged_text[0][0], pos='v')
    elif tagged_text[0][1][0].lower() == 'n':
        word = lmtzr.lemmatize(tagged_text[0][0], pos='n')
    else:
        word = tagged_text[0][0]
    return word


@jit
def remove_short_words(word):
    if len(word) < 3:
        word = ''
    return word


@jit
def remove_stop_words(word):
    """take a word and check it against the common stop words list from NLTK"""
    stops = set(stopwords.words("english"))
    if word in stops:
        word = ''
    return word


@jit
def remove_puncuation(word):
    # compile regex
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation
    word = punc_re.sub('', word)
    return word


@jit
def remove_numbers(word):
    # compile regex
    num_re = re.compile('(\\d+)')
    # remove numbers
    word = num_re.sub('', word)
    return word


def phrase_to_vec(word2vec_model, row):
    """Convert individual words to vectors and then average the vectors
    Kenter et al. 2016, "simply averaging word embeddings of all words in a text has proven to be a strong baseline or
    feature across a multitude of tasks", such as short text similarity tasks
    http://aclweb.org/anthology/P/P16/P16-1089.pdf
    Code based from https://bitbucket.org/yunazzang/aiwiththebest_byor"""
    vector_set = []
    # check if list is empty
    if row:
        for word in row:
            try:
                word_vector = word2vec_model[word]
                vector_set.append(word_vector)
            except KeyError:
                pass
    vector_set = np.mean(vector_set, axis=0).tolist()
    return vector_set


def row_text_cleaner(str_row, word2vec_model, tagger, lmtzr):
    """Returns a cleaned row after removing words not needed"""
    nlp_row = []
    for word in word_tokenize(str_row):
        nlp_row.append(remove_features(word, tagger, lmtzr))
    # unique the words
    nlp_row = list(set(nlp_row))
    phrase_vector = phrase_to_vec(word2vec_model, nlp_row)
    return phrase_vector



def text_clean(word2vec_model, df, nlp_cols):
    """Return a dataframe after simplifying text"""
    nltk.data.path.append("/opt/gmi/bd_userapps/shared/nltk_data")
    tagger = PerceptronTagger()
    lmtzr = WordNetLemmatizer()
    for col in nlp_cols:
        if col in df.columns:
            print(f'{col} in text_clean')
            df[col] = df[col].astype('str')
            df = df.replace('nan', '')
            df[col] = df[col].apply(row_text_cleaner, args=(word2vec_model, tagger, lmtzr))
    return df

