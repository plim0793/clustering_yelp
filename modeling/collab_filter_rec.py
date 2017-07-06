#### Main imports ####
import pandas as pd
import numpy as np

# sklearn
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd

# Misc.
import json

#### HELPER FUNCTIONS ####
def extract_reviews_json(file, nth=1, limit=100):
    '''
    DESCRIBE:
        - Extract the ratings/scores of reviews.
    INPUT:
        - file is the json file with the review information.
        - nth determines whether to extract consecutive lines or skip lines.
        - limit is the number of reviews to extract information from.
    OUTPUT:
        - df contains the extracted information.
    '''   
    user_list = []
    biz_list = []
    rating_list = []
    useful_list = []
    funny_list = []
    cool_list = []
    
    df = pd.DataFrame()

    with open(file) as f:
        count = 0
        for i, line in enumerate(f):
            if count % nth == 0:
                review_entry = json.loads(line)
                user_list.append(review_entry['user_id'])
                biz_list.append(review_entry['business_id'])
                rating_list.append(review_entry['stars'])
                useful_list.append(review_entry['useful'])
                funny_list.append(review_entry['funny'])
                cool_list.append(review_entry['cool'])
                
            if count > limit:
                break
            count += 1
    df['user_id'] = user_list
    df['business_id'] = biz_list
    df['stars'] = rating_list
    df['useful'] = useful_list
    df['funny'] = funny_list
    df['cool'] = cool_list
    
    return df

def extract_business_names(file, nth=1, limit=100):
    '''
    DESCRIBE:
        - Extracts the business names.
    INPUT:
        - file is the json file with the review information.
        - nth determines whether to extract consecutive lines or skip lines.
        - limit is the number of reviews to extract information from.
    OUTPUT:
        - df contains the extracted information.
    ''' 
    city_list = []
    state_list = []
    biz_encrypt_list = []
    biz_names_list = []
    
    df = pd.DataFrame()
    
    with open(file) as f:
        count = 0
        for i, line in enumerate(f):
            if count % nth == 0:
                business_entry = json.loads(line)
                
                city_list.append(business_entry['city'])
                state_list.append(business_entry['state'])
                biz_encrypt_list.append(business_entry['business_id'])
                biz_names_list.append(business_entry['name'])

            if count > limit:
                break
            count += 1
    df['city'] = city_list
    df['state'] = state_list
    df['name'] = biz_names_list
    df['encrypt'] = biz_encrypt_list
    return df

def get_recommendations(df, sample_inp, limit=10):
    '''
    DESCRIBE:
        - Get the recommendations to businesses based on user input
    INPUT:
        - df is the dataframe with the cosine distances.
        - sample_inp is the list of businesses the user is interested in.
        - limit is the number of recommendations to return.
    OUTPUT:
        - sample_sum is a dataframe with the recommendations.
    ''' 
    sample_sum = df[sample_inp].apply(lambda row: np.sum(row), axis=1)
    sample_sum = sample_sum.sort_values(ascending=False)[:limit]
    
    return sample_sum

def main():
	df_reviews = extract_reviews_json(REVIEW_FILE_PATH, nth=1, limit=2000000)
	df_names = extract_business_names(BUSINESS_FILE_PATH, nth=1, limit=144072)

	df_tot = df_reviews.merge(df_names, how="left", left_on="business_id", right_on="encrypt")
	df_tot = df_tot.drop('encrypt', axis=1)
	df_tot = df_tot.dropna()

	df_wide = pd.pivot_table(df_tot, values=['stars'],
	                                index=['name', 'user_id'],
	                                aggfunc=np.mean).unstack()

	df_wide_sample = df_wide.sample(frac=1)

	df_wide_sample = df_wide_sample.fillna(2.5)

	U, sig, VT = randomized_svd(df_wide_sample, 
	                            n_components=10,
	                            n_iter=5)

	dists = cosine_similarity(U)

	df_dists = pd.DataFrame(dists, columns=df_wide_sample.index)
	df_dists.index = df_dists.columns

	recs = get_recommendations(df_dists, sample, limit=10)

	return recs

if __name__ == '__main__':
	REVIEW_FILE_PATH = "/home/plim0793/yelp_academic_dataset_review.json"
	BUSINESS_FILE_PATH = "/home/plim0793/yelp_academic_dataset_business.json"

	sample = ['LongHorn Steakhouse', "Flury's Cafe"]

	recs = main()