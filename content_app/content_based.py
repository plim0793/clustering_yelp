# -*- coding: utf-8 -*-
import logging
import json

from flask import Flask, jsonify, render_template, redirect, url_for, send_from_directory, request, send_file
from flask_wtf import FlaskForm
from wtforms import fields
from wtforms.validators import Required, InputRequired

from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, LSHForest
from textblob import TextBlob
import spacy
import gensim
from gensim import models
from gensim.models import word2vec

import numpy as np
import pandas as pd

WTF_CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

debug = True


# create the application object
app = Flask(__name__)
app.config.from_object("config")

# unpickle objects
lsh = joblib.load('lsh.pkl')

w2v = models.KeyedVectors.load_word2vec_format("~/Documents/GoogleNews-vectors-negative300.bin.gz",binary=True)


##### FUNCTIONS AND CLASSES #####
class PredictForm(FlaskForm):
    """Fields for Predict"""
    preference = fields.TextField('Preferences:', [InputRequired()])

    submit = fields.SubmitField('Submit')

class SeparateFeaturesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_cols=None, text_cols=None):
        self.num_cols = num_cols
        self.text_cols = text_cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.num_cols:
            print("SFT: ", X.loc[:, self.num_cols].shape)
            return X.loc[:, self.num_cols]
        elif self.text_cols:
            print("SFT: ", X.loc[:, self.text_cols].shape)
            return X.loc[:, self.text_cols]
        else:
            return X

class CleanTextTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, text_col=None):
        self.text_col = text_col
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):

        X_list = X.loc[:, self.text_col].tolist()
        
        if self.text_col:
            df = pd.DataFrame()
            clean_review_list = []
            
            for review in X_list:
                clean_review = ''
                
                for word in TextBlob(review).words:
                    clean_review += word.lemmatize() + ' '
                        
                clean_review_list.append(clean_review)
                        
            df['clean_reviews'] = clean_review_list
            print("CTT: ", df.shape)
            return df
        else:
            return X

class SentimentTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_col=None):
        self.text_col = text_col
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.text_col:
            df = pd.DataFrame()
            sum_pol_list = []
            sum_sub_list = []

            for doc in X.loc[:, self.text_col]:
                sum_pol = 0
                sum_sub = 0
                doc_blob = TextBlob(doc)

                for sent in doc_blob.sentences:
                    sum_pol += sent.sentiment[0]
                    sum_sub += sent.sentiment[1]

                sum_pol_list.append(sum_pol)
                sum_sub_list.append(sum_sub)

            df['pol'] = sum_pol_list
            df['sub'] = sum_sub_list
            df['clean_reviews'] = X.loc[:, self.text_col] # Need to keep the clean reviews for the W2V transformer.
            print("ST: ", df.shape)
            return df
        else:
            return X

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_col=None, w2v=None):
        self.text_col = text_col
        self.w2v = w2v
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.text_col:
            avg_w2v_list = []
            
            for review in X.loc[:, self.text_col]:
                avg_w2v = np.zeros(300)
                count = 0
                
                for word in review:
                    try:
                        avg_w2v += w2v.word_vec(word)
                        count += 1
                    except Exception:
                        continue

                avg_w2v_list.append(avg_w2v/count)
            df = pd.DataFrame(avg_w2v_list)
#             print(df.head())
            print("W2V: ", df.shape)
            return df
        else:
            return X


def get_nearest(indices, distances, df):
    df_temp = df.loc[indices, ['rating','name','reviews']]
    df_temp['dist'] = distances
    df_temp = df_temp.sort_values(['dist'], ascending=False)
    df_temp = df_temp.reset_index()
    return df_temp



@app.route('/', methods=['GET','POST'])
def index():

	if request.method == 'POST':
		form = PredictForm(request.form)
		print('FORM: ', form)
		req = request.data.decode("utf-8")
		print(bool(req))
		if not req:
			req = "sample"
		data = json.loads(request.data.decode('utf-8'), strict=False)
		print(data)

	else:
		form = PredictForm()
		data = {'preference': 'sample preference'}

	print("PREFERENCE: ", data)

	input_pref = [data['preference']]

	print("INPUT PREF: ", input_pref)

	# Put input into a dataframe
	sample_df = pd.DataFrame(input_pref, columns=["sample"])

	# Load the pipeline
	pipe_sample = Pipeline([
                    ('split_text', SeparateFeaturesTransformer(text_cols=['sample'])),
                    ('clean', CleanTextTransformer('sample')),
                    ('sentiment', SentimentTransformer(text_col='clean_reviews')),
                    ('vectorize', Word2VecTransformer(text_col='clean_reviews', w2v=w2v))
                                        ])

	# Transform the input
	sample_transform = pipe_sample.fit_transform(sample_df)

	# Calculate distances and indices from input
	distances, indices = lsh.kneighbors(sample_transform, n_neighbors=5)

	# Get a dataframe sorted by distances
	df_sample_rec = get_nearest(indices[0], distances[0], df_rn)

	print("TOP 5: ", df_sample_rec.head())

	top_1 = df_sample_rec.loc[0, 'name']
	top_2 = df_sample_rec.loc[1, 'name']
	top_3 = df_sample_rec.loc[2, 'name']
	top_4 = df_sample_rec.loc[3, 'name']
	top_5 = df_sample_rec.loc[4, 'name']

	top_1_rev = df_sample_rec.loc[0, 'reviews']
	top_2_rev = df_sample_rec.loc[1, 'reviews']
	top_3_rev = df_sample_rec.loc[2, 'reviews']
	top_4_rev = df_sample_rec.loc[3, 'reviews']
	top_5_rev = df_sample_rec.loc[4, 'reviews']
	

	if request.method == 'POST':
		return jsonify(
					top_1 = top_1,
					top_2 = top_2,
					top_3 = top_3,
					top_4 = top_4,
					top_5 = top_5,
					top_1_rev = top_1_rev,
					top_2_rev = top_2_rev,
					top_3_rev = top_3_rev,
					top_4_rev = top_4_rev,
					top_5_rev = top_5_rev
				)


	return render_template('index.html',
							form=form,
							top_1 = top_1,
							top_2 = top_2,
							top_3 = top_3,
							top_4 = top_4,
							top_5 = top_5,
							top_1_rev = top_1_rev,
							top_2_rev = top_2_rev,
							top_3_rev = top_3_rev,
							top_4_rev = top_4_rev,
							top_5_rev = top_5_rev)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)



