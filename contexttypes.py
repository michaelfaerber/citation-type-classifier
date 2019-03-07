

import pandas as pd
from nltk.corpus import stopwords
from pprint import pprint
from time import time
import re
from collections import Counter


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from gensim.parsing import preprocessing
from gensim.utils import to_unicode
import contractions
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score, precision_score, confusion_matrix

from sklearn.base import BaseEstimator, TransformerMixin
import spacy
#nlp = spacy.load('en_core_web_md', parser=False)
nlp = spacy.load('en')

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


def read_excel(filename, sheetname):
    """ Reads and parases an Excel file, returns a data frame with the context and citationtype columns """
    # NOTE: column citation type in the files changed manually to citationType (space removed) 
    xl = pd.ExcelFile('train-500-sw.xlsx')
    df = xl.parse("Tabelle1")
    df = df[['context', 'citationType']]
    return df

def assign_citation_class(citation_type):
    """ This function takes the citation type (which might contain multiple citation types separated
    by commas and 'or' and 'and', and assign the appropriate citaion class (A, B, C, D, F))"""
    types_list = []
    if citation_type.find('ref') != -1 or citation_type.find('claim') != -1 or citation_type.find('example') != -1:
        types_list.append('F')
    if citation_type.find('concept') != -1:
        types_list.append('D')
    if citation_type.find('author') != -1:
        types_list.append('C')
    if citation_type.find('intext') != -1:
        types_list.append('B')
    if citation_type.find('incomplete') != -1 or citation_type.find('language') != -1 or citation_type.find('formula') != -1:
        types_list.append('A')
    return types_list

def remove_citation_markers(text):
    """ Removes both citation markers in parentheses with author names and years, and
    citation markers in square brackets """
    # This regex looks for parentheses, a space (or not), a set of non-digit characters followed by digits (year), and
    # then more characters (or not). It will not remove, for e.g., (this string) as it has no digits.
    regex = '\(\s*\D+\d+.*\)'    
    # This regex will match [19] [19, 20, 21] with spaces allowed in between. It will not match [asedfas, 19]. 
    # These are usually in parentheses.
    square_brackets = '\[\s*\d+[,.;\d\s]*\s*\]'
    text = re.sub(regex, '', text)
    text = re.sub(square_brackets, '', text)
    return text

def clean_text(text):
    """ Cleans the text in the only argument in various steps 
    ARGUMENTS: text: content/title, string
    RETURNS: cleaned text, string"""
    # Replace newlines by space. We want only one doc vector.
    text = text.replace('\n', ' ').lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Expand contractions: you're to you are and so on.
    text = contractions.fix(text)
    # Do not remove stop words: these are important in our case
    # text = preprocessing.remove_stopwords(text)
    # Remove punctuation -- all special characters
    text = preprocessing.strip_multiple_whitespaces(text)
    return text

def preprocessing(df):
    """ Takes a datafrane containing 2 columns (citationType and context) as input, 
    creates a citationClass column (5 classes), preprocesses the contexts and also
    removes citation markers from the contexts. """
    df['modifiedContext'] = df['context'].apply(remove_citation_markers)
    df['modifiedContext'] = df['modifiedContext'].apply(clean_text)
    df['citationClass'] = df['citationType'].apply(assign_citation_class)
    return df 

def get_counter_key(counter, key):
    """ Gets the key from the counter and if it's None, sets it to 0. """
    num_fields = counter.get(key)
    num_fields = num_fields if num_fields is not None else 0
    return num_fields
    
def get_nlp_stats(context):
    """ Uses spacy to get number of persons using ner and 4 pos tags: proper nouns, verbs, prepositions and numbers"""
    # context is a numpy array with 1 ele
    context= context[0]
    doc = nlp(context)
    person_counter = Counter(ent.label_ for ent in doc.ents if ent.label_ == 'PERSON')
    num_persons = get_counter_key(person_counter, 'PERSON')
    pos_counter = Counter(token.pos_ for token in doc if token.pos_ in (('PROPN', 'VERB', 'ADP', 'NUM')))
    num_pnouns = get_counter_key(pos_counter, 'PROPN')
    num_verbs = get_counter_key(pos_counter, 'VERB')
    num_preps = get_counter_key(pos_counter, 'ADP')
    num_numbers = get_counter_key(pos_counter, 'NUM')
    return [num_pnouns, num_verbs, num_preps, num_numbers, num_persons]

def get_features(df):
    """ Takes a data frame containing the citationClass (target) and the modifiedContext column and creates
    a set of NLP-based frequency features. Returns a new data frame with a set of feature columns and the
    modifiedContext"""
    X = df[['modifiedContext']]
    X['num_words'] = X['modifiedContext'].apply(lambda x: len(x.split()))
    num_cols = ['num_pnouns', 'num_verbs', 'num_preps', 'num_numbers', 'num_persons']
    X[num_cols] = pd.DataFrame(X['modifiedContext']).apply(get_nlp_stats,
                                                           axis=1, result_type='expand')
    return X

def get_multi_labels(df):
    """ Takes a dataframe containing the labels column citationType, and returns a set of multi label-binarized
    labels"""
    # One-hot encode the labels (5 classes), fit_transform expects lists as input: series of lists in this case
    lb = MultiLabelBinarizer(classes=['A', 'B', 'C', 'D', 'F'])
    y = lb.fit_transform(df.citationClass)
    return y

def make_ready_for_ml(excel_file, sheetname):
    """ Reads the Excel filename + sheetname taken as arguments, performs a set of preprocessing steps,
    and returns a dataframe of features X, and multi-label binarized y. """
    df = read_excel(excel_file, sheetname)
    df = preprocessing(df)
    X = get_features(df)
    y = get_multi_labels(df)
    return X, y

def create_pipeline():
    """ Applies a pipeline of selecting features, applying transformations and an an SGD model. Parameters 
    and hyperparameters have been selected in a different program. """
    
    # Pipeline to apply the TfidfVectorizer (CountVectorizer+TfidfTransformer) on the processed column of the dataframe
    tfidf_features = Pipeline([
                        ('selector', TextSelector(key='modifiedContext')),
                        ('tfidf', TfidfVectorizer(max_features= 50000, max_df=0.75, ngram_range=(1, 3),
                                                  stop_words=None))
                    ])
    
    # Pipeline to apply a Counter on each of the parts of speech in the sentence (citation context), again applied on processed
    
    '''pos_features = Pipeline([
                        ('selector', TextSelector(key='modifiedContext')),
                        ('pos', PosTagMatrix() ),
                    ])
    
    ner_features = Pipeline([
                        ('selector', TextSelector(key='modifiedContext')),
                        ('ner', NerMatrix()),
                    ])
    '''
    
    # Pipeline to apply a standard scaler on the numeric column num_words
    num_words_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_words')),
                                    ('standard', StandardScaler())
                                ])

    num_preps_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_preps')),
                                    ('standard', StandardScaler())
                                ])

    num_pnouns_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_pnouns')),
                                    ('standard', StandardScaler())
                                ])
    num_persons_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_persons')),
                                    ('standard', StandardScaler())
                                ])
    num_numbers_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_numbers')),
                                    ('standard', StandardScaler())
                                ])
    num_verbs_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_verbs')),
                                    ('standard', StandardScaler())
                                ])


    features = FeatureUnion([
                        ('tfidf', tfidf_features),
                        ('words', num_words_features),
                        ('verbs', num_verbs_features),
                        ('numbers', num_numbers_features),
                        ('persons', num_persons_features),
                        ('nouns', num_pnouns_features),
                        ('preps', num_preps_features)
        
    ])
    #print(features.get_feature_names())
    feature_engineering = Pipeline([('features', features)])
    
    pipeline = Pipeline([
        ('features_set', feature_engineering),
        #('clf', OneVsRestClassifier(SGDClassifier()))
        #('clf', OneVsRestClassifier(LinearSVC()))
        #('clf', OneVsRestClassifier(LogisticRegression()))
        #('clf', OneVsRestClassifier(BernoulliNB()))
        #('clf', OneVsRestClassifier(SVC()))
        #('clf', OneVsRestClassifier(AdaBoostClassifier()))
        #('clf', OneVsRestClassifier(RandomForestClassifier()))
        ('clf', OneVsRestClassifier(GradientBoostingClassifier()))
    ])

    return pipeline

def train(X, y):
    """ Runs a machine learning pipeline using the feature set X and the multi-labels y """
    scorers = {
    'precision_score': make_scorer(precision_score, average='weighted'),
    'recall_score': make_scorer(recall_score, average='weighted'),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='weighted')
    }
    text_pipeline = create_pipeline()
    print(text_pipeline)
    scores = cross_validate(text_pipeline, X, y, scoring=scorers,
                        cv=5, return_train_score=False)
    print scores

def main():
    """ Main function """
    X_train, y_train = make_ready_for_ml('train-500-sw.xlsx', 'Tabelle1')
    #X_test, y_test = make_ready_for_ml('test-100-nlp.xlsx', 'Tabelle1')
    train(X_train, y_train)
    # Test
    