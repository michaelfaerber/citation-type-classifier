
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from pprint import pprint
from time import time
import re
from collections import Counter
from joblib import dump, load

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelBinarizer
from sklearn.preprocessing import FunctionTransformer

from gensim import parsing
from gensim.utils import to_unicode
from gensim.models.fasttext import FastText
import contractions
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score, hamming_loss
from sklearn.metrics import precision_score, confusion_matrix, classification_report

from sklearn.base import BaseEstimator, TransformerMixin
import spacy
nlp = spacy.load('en_core_web_md', parser=False)
#nlp = spacy.load('en')
fasttextmodel = FastText.load_fasttext_format('cc.en.300.bin')

# This regex looks for parentheses, a space (or not), a set of non-digit characters followed by digits (year), and
# then more characters (or not). It will not remove, for e.g., (this string) as it has no digits.
regex_parentheses = re.compile(r'\(\s*\D+\d+.*\)')    
# This regex will match [19] [19, 20, 21] with spaces allowed in between. It will not match [asedfas, 19]. 
# These are usually in parentheses.
regex_square_brackets = re.compile(r'\[\s*\d+[,.;\d\s]*\s*\]')

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
        return X[self.key].values.reshape(-1, 1)

def get_fasttext_row_vector(row):
    return row['fasttext_vector'].tolist()

FastTextTransformer = FunctionTransformer(get_fasttext_row_vector, validate=False)

def read_excel(filename, sheetname):
    """ Reads and parases an Excel file, returns a data frame with the context and citationtype columns """
    # NOTE: column citation type in the files changed manually to citationType (space removed) 
    xl = pd.ExcelFile(filename)
    df = xl.parse(sheetname)
    df = df[['context', 'citationType']]
    # remove control sequences from contexts
    # https://stackoverflow.com/questions/2296525/how-to-encode-decode-escape-sequence-characters-in-python
    df.context = df.context.apply(lambda s: re.sub('[\x00-\x08\x0B-\x1F]', '', s))
    return df

def assign_citation_class(citation_type):
    """ This function takes the citation type (which might contain multiple citation types separated
    by commas and 'or' and 'and', and assign the appropriate citaion class (A, B, C, D, F))"""
    types_list = []
    if citation_type.find('ref') != -1 or citation_type.find('claim') != -1 or citation_type.find('example') != -1:
        types_list.append('E')
    if citation_type.find('concept') != -1:
        types_list.append('D')
    if citation_type.find('author') != -1:
        types_list.append('C')
    if citation_type.find('intext') != -1:
        types_list.append('B')
    if citation_type.find('incomplete') != -1 or citation_type.find('language') != -1 or citation_type.find('formula') != -1:
        types_list.append('A')
    return types_list

def replace_citation_markers(text):
    """ Removes both citation markers in parentheses with author names and years, and
    citation markers in square brackets, and adds a <citationmarker> marker in their place. """
    text = regex_parentheses.sub(' <citationmarker> ', text)
    #print(text)
    text = regex_square_brackets.sub(' <citationmarker> ', text)
    #print(text)
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
    text = parsing.preprocessing.strip_multiple_whitespaces(text)
    return text

def preprocessing(df):
    """ Takes a datafrane containing 2 columns (citationType and context) as input, 
    creates a citationClass column (5 classes), preprocesses the contexts and also
    removes citation markers from the contexts. """
    df['modifiedContext'] = df['context'].apply(replace_citation_markers)
    df['modifiedContext'] = df['modifiedContext'].apply(clean_text)
    df['citationClass'] = df['citationType'].apply(assign_citation_class)
    return df 

def get_counter_key(counter, key):
    """ Gets the key from the counter and if it's None, sets it to 0. """
    num_fields = counter.get(key)
    num_fields = num_fields if num_fields is not None else 0
    return num_fields
    
def get_nearest_pos(citation_positions, pos_positions):
    """ Gets nearest pos (based on pos_positions) for each of the citation markers and 
    returns a sum of the list of the nearest pos for each of the markers."""
    x= []
    y= []
    if pos_positions == [] or citation_positions == 0:
        return 0
    for markerpos in citation_positions:
        for posposition in pos_positions:
            x.append(abs(markerpos-posposition))
        
        y.append(min(x))
        x = []
    # Return sum without normalizing for the number of citations.
    return sum(y)

def get_nlp_stats(context):
    """ Uses spacy to get number of persons using ner and 4 pos tags: proper nouns, verbs, prepositions and numbers"""
    # context is a numpy array with 1 ele
    context= context[0]
    doc = nlp(context)
    context_words = parsing.preprocessing.strip_multiple_whitespaces(
        parsing.preprocessing.strip_punctuation(context)).split()
    num_words = len(context_words)
    person_counter = Counter(ent.label_ for ent in doc.ents if ent.label_ == 'PERSON')
    num_persons = get_counter_key(person_counter, 'PERSON') / num_words
    pos_counter = Counter(token.pos_ for token in doc if token.pos_ in (('PROPN', 'VERB', 'ADP', 'NUM', 'NOUN', 'PUNCT')))
    # noemalized counters pos
    num_pnouns = get_counter_key(pos_counter, 'PROPN') / num_words
    num_nouns = get_counter_key(pos_counter, 'NOUN') / num_words
    num_verbs = get_counter_key(pos_counter, 'VERB') / num_words
    num_preps = get_counter_key(pos_counter, 'ADP') / num_words
    num_numbers = get_counter_key(pos_counter, 'NUM') / num_words
    num_punctuations = get_counter_key(pos_counter, 'PUNCT') / num_words
    num_citation_markers = context.count('<citationmarker>')
    # Create a new feature which is the average (normalized) position of citation markers in the context.
    # E.g: this is the <citation> here and another <citation> here. 
    # num_words = 9, 2 citation markers: 4/9, 8/9. Result = 4/9 + 8/9 = 12/9. If there is only one marker, this will
    # be less than 1. No need to normalize. This feature, when combined with the num_contexts feature, gives the whole picture.
    citation_positions = [index for index, word in enumerate(context_words) if word == 'citationmarker']
    # a/b + c/b = (a+c)/b
    citation_positions_sum = sum(citation_positions) / num_words 
    # Again, use spacy to get the pos of the positions in a list
    stripped_doc = nlp(' '.join(context_words))
    pos_list = [word.pos_ for word in stripped_doc]
    entities_list = [ent.label_ for ent in stripped_doc.ents]
    #print(pos_list)
    personindices = [index for index, entity in enumerate(entities_list) if entity == 'PERSON'] 
    verbindices = [index for index, pos in enumerate(pos_list) if pos == 'VERB']
    nounindices = [index for index, pos in enumerate(pos_list) if pos == 'NOUN']
    pnounindices = [index for index, pos in enumerate(pos_list) if pos == 'PROPN']
    prepindices = [index for index, pos in enumerate(pos_list) if pos == 'ADP']
    #persons = [index for index, word in enumerate(context_words) if pos == 'VERB']
    nearest_verb_positions_sum = get_nearest_pos(citation_positions, verbindices)
    nearest_noun_positions_sum = get_nearest_pos(citation_positions, nounindices)
    #nearest_person_positions = get_nearest_pos(citation_positions, nounindices)
    nearest_pnoun_positions_sum = get_nearest_pos(citation_positions, pnounindices)
    nearest_prep_positions_sum = get_nearest_pos(citation_positions, prepindices)
    nearest_person_positions_sum = get_nearest_pos(citation_positions, personindices)
    fasttext_vector = get_fasttext_vector(context_words)
    return [num_words, num_nouns, num_pnouns, num_punctuations, num_verbs, num_preps, num_numbers, num_persons,  
            num_citation_markers, citation_positions_sum, nearest_verb_positions_sum, nearest_noun_positions_sum,
            nearest_pnoun_positions_sum, nearest_prep_positions_sum, nearest_person_positions_sum, fasttext_vector]

def get_fasttext_word_vector(word):
    """ Gets the vector for a word. If all ngrams of the word are not seen, it returns a vector of 0s"""
    # vector size=300
    try:
        return fasttextmodel.wv[word]
    except KeyError:
        return np.zeros(300)

def get_fasttext_vector(context_words):
    """ Gets fast text vectors for each of the context words and returns their ele-wise avg"""
    return np.mean([get_fasttext_word_vector(word) for word in context_words], axis=0)

def get_features(df):
    """ Takes a data frame containing the citationClass (target) and the modifiedContext column and creates
    a set of NLP-based frequency features. Returns a new data frame with a set of feature columns and the
    modifiedContext"""
    X = df[['modifiedContext']]
    X_columns = ['num_words', 'num_nouns',  'num_pnouns', 'num_punctuations', 'num_verbs', 'num_preps', 'num_numbers', 
             'num_persons', 'num_citation_markers', 'citation_positions_sum', 'nearest_verb_positions_sum',
             'nearest_noun_positions_sum', 'nearest_pnoun_positions_sum', 'nearest_prep_positions_sum',
             'nearest_person_positions_sum', 'fasttext_vector']
    X[X_columns] = pd.DataFrame(X['modifiedContext']).apply(get_nlp_stats, axis=1, result_type='expand')
    # Make the num_columns float to prevent StandardScaler grumbling about Datatypeconversion during training.
    # fasttext vector will not be sent to the StandardScaler, don't touch it.
    X[X_columns[:-1]] = X[X_columns[:-1]].astype(float)
    return X

def get_multi_labels(df):
    """ Takes a dataframe containing the labels column citationType, and returns a set of multi label-binarized
    labels"""
    # One-hot encode the labels (5 classes), fit_transform expects lists as input: series of lists in this case
    lb = MultiLabelBinarizer(classes=['A', 'B', 'C', 'D', 'E'])
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
    num_citation_markers_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_citation_markers')),
                                    ('standard', StandardScaler())
                                ])
    citation_positions_features = Pipeline([
                                    ('selector', NumberSelector(key='citation_positions_sum')),
                                    ('standard', StandardScaler())
                                ])
    nearest_verb_features = Pipeline([
                                    ('selector', NumberSelector(key='nearest_verb_positions_sum')),
                                    ('standard', StandardScaler())
                                ])
    nearest_noun_features = Pipeline([
                                    ('selector', NumberSelector(key='nearest_noun_positions_sum')),
                                    ('standard', StandardScaler())
                                ])
    nearest_pnoun_features = Pipeline([
                                    ('selector', NumberSelector(key='nearest_pnoun_positions_sum')),
                                    ('standard', StandardScaler())
                                ])
    nearest_prep_features = Pipeline([
                                    ('selector', NumberSelector(key='nearest_prep_positions_sum')),
                                    ('standard', StandardScaler())
                                ])
    nearest_person_features = Pipeline([
                                    ('selector', NumberSelector(key='nearest_person_positions_sum')),
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
    num_nouns_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_nouns')),
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
    num_punctuations_features =  Pipeline([
                                    ('selector', NumberSelector(key='num_punctuations')),
                                    ('standard', StandardScaler())
                                ])
    # https://stackoverflow.com/questions/47745288/how-to-featureunion-numerical-and-text-features-in-python-sklearn-properly
    #fasttext_features = Pipeline([
    #                                ('selector', NdarraySelector(key='fasttext_vector'))#,
                                    #('standard', StandardScaler())
    #                           ])
    features = FeatureUnion([
                        ('tfidf', tfidf_features),
                        ('words', num_words_features),
                        ('verbs', num_verbs_features),
                        ('numbers', num_numbers_features),
                        ('persons', num_persons_features),
                        ('pnouns', num_pnouns_features),
                        ('nouns', num_nouns_features),
                        ('punctuations', num_punctuations_features),
                        ('preps', num_preps_features),
                        ('citations', num_citation_markers_features),
                        ('citationpositions', citation_positions_features),
                        ('nearestverbs', nearest_verb_features),
                        ('nearestnouns', nearest_noun_features),
                        ('nearestpnouns', nearest_pnoun_features),
                        ('nearestpreps', nearest_prep_features),
                        ('nearestpersons', nearest_person_features),
                        #('fasttext', NdarraySelector(key='fasttext_vector'))#,
                        ('fasttext', FastTextTransformer)       
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
        #('clf', OneVsRestClassifier(AdaBoostClassifier(random_state=13)))
        #('clf', OneVsRestClassifier(RandomForestClassifier(random_state=13)))
        ('clf', OneVsRestClassifier(GradientBoostingClassifier(random_state=13)))
    ])
    return pipeline

def train(X, y, cv=False):
    """ Runs a machine learning pipeline using the feature set X and the multi-labels y """
    scorers = {
    'precision_score': make_scorer(precision_score, average='weighted'),
    'recall_score': make_scorer(recall_score, average='weighted'),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='weighted')
    }
    text_pipeline = create_pipeline()
    print(text_pipeline)
    if cv:
        scores = cross_validate(text_pipeline, X, y, scoring=scorers,
                                cv=5, return_train_score=False)
        #dump(text_pipeline, 'cittype_pipeline.joblib')
        print(scores)
    else:
        text_pipeline.fit(X, y)
        dump(text_pipeline, 'cittype_pipeline.joblib')

def calculate_metrics(y_test, y_pred, ml_model):
    """ Calculates a number of metrics using the model, the predicted y and the true y.
    ARGUMENTS: y_test: test (validation) set labels, Pandas Series
               y_pred: predicted labels, Pandas Series
               ml_model: sklearn model (hyperparams printed in log file)
               val_filetype: string 'Buzzfeed Validation File' or
               'Crowdsourced File used as a validation file'
    RETURNS: None"""
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    hamming_loss_score = hamming_loss(y_test, y_pred)
    pprint(classification_report(y_test, y_pred))
    print("Accuracy=", accuracy, '\n')
    print("Hamming loss=", hamming_loss_score, '\n')

def test(X, y):
    """ """
    model = load('cittype_pipeline.joblib')
    y_pred = model.predict(X)
    # Sometimes, it doesn't predict any labels. 
    # Get sum of all the y_pred, rows which have sum=0 have no class predicted
    sums = np.sum(y_pred, axis=1)
    # Prob of all 0 predictions: Threshold seems to be around 0.49/0.5
    y_pred_prob = model.predict_proba(X)
    # Get all 0 sum indices (where returns a tuple)
    index_all_zeros = np.where(sums==0)[0]
    max_labels = np.argmax(y_pred_prob[index_all_zeros], axis=1)
    # one hot encode
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(max_labels)+1))
    b = label_binarizer.transform(max_labels)
    y_pred[index_all_zeros] = b 
    calculate_metrics(y, y_pred, model)

def main():
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("traintest", choices=["train", "test"],
        help="Train or test?")
    args=parser.parse_args()
    print(args.traintest)
    if args.traintest == 'train':
        X_train, y_train = make_ready_for_ml('train-500-sw.xlsx', 'Tabelle1')
        #train(X_train, y_train, cv=True)
        train(X_train, y_train)
    # Test
    elif args.traintest == 'test':
        X_test, y_test = make_ready_for_ml('test-100-nlp.xlsx', 'Tabelle1')
        test(X_test, y_test)

if __name__ == '__main__':
    main()
