#### Main imports ####
import pandas as pd
import numpy as np
import transformers

# sklearn
from sklearn.externals import joblib
from sklearn import pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LSHForest

# NLP 
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
import gensim
from gensim import models
from gensim.models import word2vec
import snowballstemmer

# Misc.
import re
import datetime
import time
import logging
import math

% matplotlib inline

sns.set_style("white")
sns.set_style('ticks')
sns.set_style({'xtick.direction': u'in', 'ytick.direction': u'in'})
sns.set_style({'legend.frameon': True})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.FileHandler('logging_records.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

#### HELPER FUNCTIONS ####
def split_dataframe():
    '''
    DESCRIBE:
        - Split the dataframe by the word2vector columns and the ratings/names columns.
    INPUT:
        - No input.
    OUTPUT:
        - df_rn is the dataframe with the name, rating, and review text.
        - df_300 is the dataframe with the 300 features from the word2vector model.
    '''    
    df = joblib.load(PATH_TO_DF)
    df_rn = df[['name', 'rating', 'reviews']]
    df_rn['name'] = df_rn['name'].apply(lambda x: re.sub('[0-9]*_', '', x))
    
    df_300 = df[[i for i in range(1,301)]]
    
    return df_rn, df_300

def modeling(df, model_obj):
    '''
    DESCRIBE:
        - Fit the model with the dataframe.
    INPUT:
        - df is the dataframe with data.
        - model_obj is the instantiated model object.
    OUTPUT:
        - model is the trained model object.
    '''  
    model = model_obj
    model.fit(df)
    
    return model

def get_nearest(indices, distances, df):
    '''
    DESCRIBE:
        - Sorts the closest reviews based on the distances.
    INPUT:
        - indices and distances are the values that determine how to sort the dataframe.
        - df contains the data that is to be analyzed.
    OUTPUT:
        - df_temp is the sorted dataframe.
    '''  
    df_temp = df.loc[indices, ['rating','name','reviews']]
    df_temp['dist'] = distances
    df_temp = df_temp.sort_values(['dist'], ascending=False)
    df_temp = df_temp.drop_duplicates()
    df_temp = df_temp.reset_index()
    return df_temp

def get_recommendations(inp_str, pipeline, model, df, num_rec=20):
    '''
    DESCRIBE:
        - Logs the top recommendations based on the input string.
    INPUT:
        - inp_str is the preferences inputted by the user.
        - pipeline is the pipeline that transforms the user input into the same vector space as the model.
        - model is the trained model.
        - df is the dataframe with the data to be analyzed.
        - num_rec is the number of recommendations to log.
    OUTPUT:
        - df_sample_rec is the dataframe that contains the recommendations.
    '''  
    if isinstance(inp_str, str):
        input_as_list = [inp_str]
        sample_df = pd.DataFrame(input_as_list, columns=["sample"])
        sample_transform = pipeline.fit_transform(sample_df)
        
        dist, indices = model.kneighbors(sample_transform, n_neighbors=num_rec)
        df_sample_rec = get_nearest(indices[0], dist[0], df)
        
        for i in range(len(df_sample_rec)):
            logging.info(str(df_sample_rec.loc[i, 'name']))
            logging.info("NUMBER " + str(i) + ": ", str(df_sample_rec.loc[i, 'reviews']))
            logging.info("\n")
        
        return df_sample_rec
    else:
        logging.warning('Input was not a string')
        return False

def main():
	df_rn, df_300 = split_dataframe()
	df_sample_rec = get_recommendations(sample_input, PIPELINE, model, df_rn, num_rec=20)

	return df_sample_rec

if __name__ == '__main__':

	# ONLY RUN ONCE AT THE START OF THE KERNEL
	w2v = models.KeyedVectors.load_word2vec_format("~/Documents/GoogleNews-vectors-negative300.bin.gz",binary=True)
	PATH_TO_DF = '../data/df_out'

	sample_input = "good coffee and quiet setting and fast wifi"

	PIPELINE = Pipeline([
	    ('split_text', transformers.SeparateFeaturesTransformer(text_cols=['sample'])),
	    ('clean', transformers.CleanTextTransformer('sample')),
	    ('sentiment', transformers.SentimentTransformer(text_col='clean_reviews')),
	    ('vectorize', transformers.Word2VecTransformer(text_col='clean_reviews', w2v=w2v))
	])

	df_sample_rec = main()
