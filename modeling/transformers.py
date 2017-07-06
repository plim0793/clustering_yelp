from sklearn.base import BaseEstimator, TransformerMixin


class DataframeToSeriesTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Transforms a dataframe object to a series object.
    INPUT:
        - col is the column name that needs to be made into a series object.
    OUTPUT:
        - X is the series object.
    '''
    def __init__(self, col=None):
        self.col = col
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.col:
            logger.info("DTST: {}".format(str(X[self.col].shape)))
            return X[self.col]
        else:
            return X
        
class SeparateFeaturesTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Separates the features in a dataframe into dataframes with strictly numerical or text features.
    INPUT:
        - num_cols is the list of numerical feature names.
        - text_cols is the list of text feature names.
    OUTPUT:
        - X is the dataframe object.
    '''
    def __init__(self, num_cols=None, text_cols=None):
        self.num_cols = num_cols
        self.text_cols = text_cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.num_cols:
            logger.info("SFT: {}".format(str(X.loc[:, self.num_cols].shape)))
            return X.loc[:, self.num_cols]
        elif self.text_cols:
            logger.info("SFT: {}".format(str(X.loc[:, self.num_cols].shape)))
            return X.loc[:, self.text_cols]
        else:
            return X
        
class CleanTextTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Cleans the text feature in a dataframe.
    INPUT:
        - text_col is the text feature name.
    OUTPUT:
        - X is the dataframe object.
    '''
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
            logger.info("CTT: {}".format(str(df.shape)))
            return df
        else:
            return X
        
class DensifyTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Transforms a sparse array to a dense array.
    INPUT:
        - No other input other than self.
    OUTPUT:
        - df is the dataframe object.
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X.toarray())
        logger.info("DT: {}".format(str(df.shape)))
        return df
    
class SentimentTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Uses sentiment analysis to create a polarity and subjectivity score.
    INPUT:
        - text_col is the name of the text column.
    OUTPUT:
        - df is the dataframe object.
    '''
    def __init__(self, text_col=None):
        self.text_col = text_col
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.text_col:
            df = pd.DataFrame()
            sum_pol_list = []
            sum_sub_list = []
            doc_list = X.loc[:, self.text_col].tolist()

            for doc in doc_list:
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
            logger.info("ST: {}".format(str(df.shape)))
            return df
        else:
            return X

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Use the trained Word2Vec model to create the 300 dimensional matrix.
    INPUT:
        - text_col is the name of the text column.
    OUTPUT:
        - df is the dataframe object.
    '''    
    def __init__(self, text_col=None, w2v=None):
        self.text_col = text_col
        self.w2v = w2v
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.text_col:
            avg_w2v_list = []
            review_list = X.loc[:, self.text_col].tolist()
            
            for review in review_list:
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
            logger.info("W2V: {}".format(str(df.shape)))
            return df
        else:
            return X
        
class ToDataFrameTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Create a dataframe object from an array.
    INPUT:
        - No other input other than self.
    OUTPUT:
        - df is the dataframe object.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X)
        logger.info("TDFT: {}".format(str(df.shape)))
        return df
        
class DropTextTransformer(BaseEstimator, TransformerMixin):
    '''
    DESCRIBE:
        - Drops the text columns from the dataframe.
    INPUT:
        - text_col is the name of the text column.
    OUTPUT:
        - df is the dataframe object.
    '''
    def __init__(self, text_col=None):
        self.text_col = text_col
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        if self.text_col:
            df = X.drop(self.text_col, axis=1)
            logger.info("DTT: {}".format(str(df.shape)))
            return df
