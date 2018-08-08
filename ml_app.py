from __future__ import division, print_function
from flask import Flask, request, render_template
from utils import tokenize  # tokenizer used when training TFIDF vectorizer
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# coding=utf-8

from keras import backend as K
# K.clear_session()

# import examples.example_helper
""" Module import helper.
Modifies PATH in order to allow us to import the deepmoji directory.
"""
import sys
import os
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

import glob
import re

# Keras
from keras import backend as K
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
basepath = os.path.abspath("./toxic_comment_classifier_web_app")  # important for server to find models folder
basepath = os.path.abspath(".")  # important for server to find models folder
models_directory = 'models'


@app.before_first_request
def nbsvm_models():
#     # from utils import tokenize

     global tfidf_model
     global logistic_identity_hate_model
     global logistic_insult_model
     global logistic_obscene_model
     global logistic_severe_toxic_model
     global logistic_threat_model
     global logistic_toxic_model

with open(basepath + '/models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
    tfidf_model = pickle.load(tfidf_file)

with open(basepath + '/models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
    logistic_toxic_model = pickle.load(logistic_toxic_file)
with open(basepath + '/models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
    logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)
with open(basepath + '/models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
    logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)
with open(basepath + '/models/logistic_insult.pkl', 'rb') as logistic_insult_file:
    logistic_insult_model = pickle.load(logistic_insult_file)
with open(basepath + '/models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
    logistic_obscene_model = pickle.load(logistic_obscene_file)
with open(basepath + '/models/logistic_threat.pkl', 'rb') as logistic_threat_file:
    logistic_threat_model = pickle.load(logistic_threat_file)

maxlen = 30
batch_size = 32

emo = ['😂', '😒', '😩', '😭', '😍',
       '😔', '👌', '😊', '❤', '😏',
       '😁', '🎶', '😳', '💯', '😴',
       '😌', '☺', '🙌', '💕', '😑',
       '😅', '🙏', '😕', '😘', '♥',
       '😐', '💁', '😞', '🙈', '😫',
       '✌', '😎', '😡', '👍', '😢',
       '😪', '😋', '😤', '✋', '😷',
       '👏', '👀', '🔫', '😣', '😈',
       '😓', '💔', '💓', '🎧', '🙊',
       '😉', '💀', '😖', '😄', '😜',
       '😠', '🙅', '💪', '👊', '💜',
       '💖', '💙', '😬', '✨']

emoToColor = ['191, 255, 0', '0, 96, 128', '0, 153, 204', '0, 0, 153', '255, 132, 102',
              '77, 255, 210', '255, 177, 0', '255, 132, 102', '230, 0, 0', '255, 147, 255',
              '255, 132, 102', '255, 51, 204', '191, 255, 0', '255, 255, 0', '147, 166, 89',
              '204, 51, 255', '255, 64, 0', '255, 177, 0', '255, 64, 0', '0, 96, 128',
              '191, 255, 0', '230, 0, 0', '0, 96, 128', '255, 64, 0', '230, 0, 0',
              '0, 96, 128', '204, 51, 255', '77, 255, 210', '255, 30, 98', '0, 153, 204',
              '255, 255, 0', '204, 51, 255', '99, 0, 77', '255, 177, 0', '0, 0, 153',
              '77, 255, 210', '255, 132, 102', '99, 0, 77', '255, 255, 0', '147, 166, 89',
              '255, 177, 0', '255, 30, 98', '147, 166, 89', '77, 255, 210', '204, 51, 255',
              '77, 255, 210', '0, 0, 153', '230, 0, 0', '255, 51, 204', '255, 30, 98',
              '255, 147, 255', '191, 255, 0', '0, 153, 204', '255, 132, 102', '255, 147, 255',
              '99, 0, 77', '255, 255, 0', '255, 255, 0', '255, 255, 0', '255, 64, 0',
              '255, 64, 0', '255, 64, 0', '191, 255, 0', '255, 64, 0'
              ]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

# print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()
model._make_predict_function()

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

def model_predict(TEST_SENTENCES):
    print(TEST_SENTENCES)

    # print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    print (tokenized)
    prob = model.predict_function([tokenized])[0]
    return prob

import pdb

def get_emoji(TEST_SENTENCES):
    # K.clear_session()
    # Find top emojis for each sentence. Emoji ids (0-63)
    # correspond to the mapping in emoji_overview.png
    # at the root of the DeepMoji repo.
    # print('Writing results to {}'.format(OUTPUT_PATH))
    scores = []
    t_score = []
    print (TEST_SENTENCES)
    prob = model_predict(TEST_SENTENCES)
    t_score.append(TEST_SENTENCES[0])
    t_prob = prob[0]
    ind_top = top_elements(t_prob, 4)
    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)
    print(t_score)
    return t_score

@app.route('/', methods=['GET'])
def my_form():
    return render_template('hatespeech.html')


@app.route('/', methods=['POST'])
def my_form_post():
    """
        Takes the comment submitted by the user, apply TFIDF trained vectorizer to it, predict using trained models
    """

    text = request.form['text']
    temp = []
    temp.append(request.form['text'])
    length = len(temp[0].split())

    comment_term_doc = tfidf_model.transform([text])

    dict_preds = {}

    dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)
        if length > 1:
            result = get_emoji(temp)
            norm = float(result[6]) + float(result[7]) + float(result[8]) + float(result[9])
            inx1 = int(result[2])
            inx2 = int(result[3])
            inx3 = int(result[4])
            inx4 = int(result[5])
            value = emo[inx1]+emo[inx2]+emo[inx3]+emo[inx4]+'\n'+str(result[0])
            color1 = "rgba(" + emoToColor[inx1] + ', ' + str(float(result[6]) / norm) + ")"
            color2 = "rgba(" + emoToColor[inx2] + ', ' + str(float(result[7]) / norm) + "), rgba(" + emoToColor[inx2] + ",  0.0)"
            color3 = "rgba(" + emoToColor[inx3] + ', ' + str(float(result[8]) / norm) + "), rgba(" + emoToColor[inx3] + ",  0.0)"
            color4 = "rgba(" + emoToColor[inx4] + ', ' + str(float(result[9]) / norm) + "), rgba(" + emoToColor[inx4] + ",  0.0)"
            print (color1, color2, color3, color4)
    return render_template('hatespeech.html', text=text, result=value, color1=color1, color2=color2, color3=color3, color4=color4,
                            pred_toxic=dict_preds['pred_toxic'],
                           pred_severe_toxic=dict_preds['pred_severe_toxic'],
                           pred_identity_hate=dict_preds['pred_identity_hate'],
                           pred_insult=dict_preds['pred_insult'],
                           pred_obscene=dict_preds['pred_obscene'],
                           pred_threat=dict_preds['pred_threat'])

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)