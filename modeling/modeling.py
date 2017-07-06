#### Main imports ####
import pandas as pd
import numpy as np
import transformers

# sklearn
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score

# NLP 
from textblob import TextBlob
from gensim import models

# Misc.
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = logging.FileHandler('logging_records.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

#### HELPER FUNCTIONS ####
def clean_df():
    '''
    DESCRIBE:
        - Preprocesses the data.
    INPUT:
        - df is the dataframe that needs to be cleaned.
    OUTPUT:
        - The dataframe that is outputted has the columns reordered and data types changed.
    '''
    if os.path.isfile(PATH_TO_DATA):
        df = joblib.load(PATH_TO_DATA)
    else:
        logger.warning("Invalid path to data")
        return False
    
    df = df[['name', 'rating' ,'reviews']]
    df['rating'] = df['rating'].apply(lambda x: int(x))
    return df

def fit_model(pipe, model, df_orig):
    '''
    DESCRIBE:
        - Fit the model through the pipeline and get scoring metrics for the model.
    INPUT:
        - pipe is the pipeline to run the data through.
        - model is the model object that will be used to fit the data.
        - df_orig is the data.
    OUTPUT:
        - df_transformed is the dataframe that is outputted from the pipeline.
        - pred is the predictions for the data for this particular model.
    '''
    df_transformed = pipe.fit_transform(df_orig)
    pred = model.fit_predict(df_transformed)
    print("Number of Clusters: ", len(np.unique(pred)))
    if len(np.unique(model.labels_)) > 1:
        logger.info("Silhouette Coefficient: %0.3f" % silhouette_score(df_transformed, model.labels_))
    return df_transformed, pred

def model_metrics(model_dict, pipe, df):
    '''
    DESCRIBE:
        - Fits a dictionary of models through the pipeline and get scoring metrics for the models.
    INPUT:
        - pipe is the pipeline to run the data through.
        - model_dict is a dictionary of the model objects that will be used to fit the data.
        - df is the data.
    OUTPUT:
        - model_dfs contains the transformed dataframe and scoring metric for each model.
    '''
    model_dfs = {}
    for name, model in model_dict.items():
        print(name)
        temp_df, temp_score = fit_model(pipe, model, df)
        model_dfs[name] = [temp_df, temp_score]
    return model_dfs

def add_feature_space(df, transformed_df, path):
    '''
    DESCRIBE:
        - Adds the feature space that was produced by the W2V model to the original dataframe.
    INPUT:
        - df is the data.
        - transformed_df is the dataframe containing the transformed data.
        - path where the dataframe should be saved.
    OUTPUT:
        - df_out is the dataframe with the new features appended.
    '''
    df_out = pd.DataFrame(transformed_df, columns=['rating'] + [i for i in range(1,301)])
    df_out['name'] = df_out['name'].tolist()
    df_out['reviews'] = df_out['reviews'].tolist()
    
    if not os.path.isdir(path):
        os.mkdir(path)

    joblib.dump(df_out, os.path.join(path, str(df_out)))
    
    return df_out

def main():
	df = clean_df()

	pipe = Pipeline([
	    ('text_feat', Pipeline([
	        ('split_text', transformers.SeparateFeaturesTransformer(text_cols=['reviews'])),
	        ('clean', transformers.CleanTextTransformer('reviews')),
	        ('sentiment', transformers.SentimentTransformer(text_col='clean_reviews')),
	        ('vectorize', transformers.Word2VecTransformer(text_col='clean_reviews', w2v=w2v))
	    ]))
	])

	model_dict = {
	    "agg": AgglomerativeClustering(n_clusters=5,
	                                   affinity="cosine",
	                                   linkage="complete"),
	    "birch": Birch(threshold=0.5,
	                   n_clusters=5,
	                   branching_factor=50),
	    "db": DBSCAN(eps=0.5,
	                 min_samples=10,
	                 metric="euclidean")
	}

	model_metrics_dict = model_metrics(model_dict, pipe, df)
	df_out = add_feature_space(df, model_metrics_dict_W2V['birch'][0], '../data/')

	return df_out

if __name__ == '__main__':
	PATH_TO_DATA = '../data/df_tot'
	w2v = models.KeyedVectors.load_word2vec_format("~/Documents/GoogleNews-vectors-negative300.bin.gz",binary=True)

	df_out = main()
