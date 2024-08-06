import nltk
import spacy
import collections
import textstat
import transformers
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import numpy as np

print("Transformers version: ", transformers.__version__)
import pandas as pd
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
# tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)
# import datasets
# from datasets import load_dataset
# from datasets import Dataset
nlp = spacy.load('en_core_web_sm')
sw_spacy = nlp.Defaults.stop_words

def count_noun(x):
    res = [token.pos_ for token in nlp(x)]
    return collections.Counter(res)['NOUN']

def count_verb(x):
    res = [token.pos_ for token in nlp(x)]
    return collections.Counter(res)['VERB']

def count_adj(x):
    res = [token.pos_ for token in nlp(x)]
    return collections.Counter(res)['ADJ']

def count_adv(x):
    res = [token.pos_ for token in nlp(x)]
    return collections.Counter(res)['ADV']

#This function is used to calcualte the sum of affective valence scores of all tokens in a text.  
def Affective_Valence_Score(x):
    try:
        df = pd.read_csv(r'affective_all.csv',delimiter=',',encoding='latin-1')        
        res=0
        token = word_tokenize(x)                
        for word in token:                
            df1=(df['Valence Mean'].loc[df['Description'] == word])                       
            for line in list(df1):
                res+=line                                
    except:
        print('Exception occurs in Affective_Valence_Score function')
    return res


#This function is used to calcualte the sum of affective arousal scores of all tokens in a text.  
def Affective_Arousal_Score(x):
    try:
        df = pd.read_csv(r'affective_all.csv',delimiter=',',encoding='latin-1')        
        res=0
        token = word_tokenize(x)                
        for word in token:                
            df1=(df['Arousal Mean'].loc[df['Description'] == word])                       
            for line in list(df1):
                res+=line                                
    except:
        print('Exception occurs in Affective_Arousal_Score function')
    return res
                
# This function is used to calcualte the sum of affective dominance scores of all tokens in a text.  
def Affective_Dominance_Score(x):
    try:
        df = pd.read_csv(r'affective_all.csv',delimiter=',',encoding='latin-1')        
        res=0
        token = word_tokenize(x)                
        for word in token:                
            df1=(df['Dominance Mean'].loc[df['Description'] == word])                       
            for line in list(df1):
                res+=line                                
    except:
        print('Exception occurs in Affective_Dominance_Score function')
    return res

def extract_features(df, title):
    # domain-specific features
    feature_vector = []
    
    # sentiment features
    df['polarity'] = df[title].apply(lambda text: TextBlob(text).sentiment.polarity)
    df['subjectivity'] = df[title].apply(lambda text: TextBlob(text).sentiment.subjectivity)
    df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row[title]), axis=1)
    pos_word_list=[]
    for word in df['tokenized_text']:
        for token in word:
            if TextBlob(token).sentiment.polarity >= 0.5:
                pos_word_list.append(token)
    pattern = '|'.join(pos_word_list)
    df['positive_word']= df[title].str.count(pattern)
    Neg_word_list=[]
    for word in df['tokenized_text']:
        for token in word:
            if TextBlob(token).sentiment.polarity <= -0.5:
                Neg_word_list.append(token)
    pattern_1 = '|'.join(Neg_word_list)
    df['negative_word']= df[title].str.count(pattern_1)

    # affective features
    df['valence'] = df[title].apply(Affective_Valence_Score)
    df['dominance'] = df[title].apply(Affective_Dominance_Score)
    df['arousal'] = df[title].apply(Affective_Arousal_Score)

    # syntactic features
    df['noun'] = df[title].apply(count_noun)
    df['verb'] = df[title].apply(count_verb)
    df['adj'] = df[title].apply(count_adj)
    df['adv'] = df[title].apply(count_adv)

    # readability features
    df['Flesch_reading_ease'] = df[title].apply(lambda text: textstat.flesch_reading_ease(text))
    df['Gunning_fog'] = df[title].apply(lambda text: textstat.gunning_fog(text))
    df['Automated_readability_index'] = df[title].apply(lambda text: textstat.automated_readability_index(text))
    df['Coleman_liau_index'] = df[title].apply(lambda text: textstat.coleman_liau_index(text))
    df['Syllable_count'] = df[title].apply(lambda text: textstat.syllable_count(text))

    for i, x in df.iterrows():
        feature_vector.append([])
        feature_vector[i].append(x['polarity'])
        feature_vector[i].append(x['subjectivity'])
        feature_vector[i].append(x['positive_word'])
        feature_vector[i].append(x['negative_word'])
        feature_vector[i].append(x['valence'])
        feature_vector[i].append(x['dominance'])
        feature_vector[i].append(x['arousal'])
        feature_vector[i].append(x['noun'])
        feature_vector[i].append(x['verb'])
        feature_vector[i].append(x['adj'])
        feature_vector[i].append(x['adv'])
        feature_vector[i].append(x['Flesch_reading_ease'])
        feature_vector[i].append(x['Gunning_fog'])
        feature_vector[i].append(x['Automated_readability_index'])
        feature_vector[i].append(x['Coleman_liau_index'])
        feature_vector[i].append(x['Syllable_count'])
        
    features_np = np.array(feature_vector)
    return features_np.shape[1], features_np
        