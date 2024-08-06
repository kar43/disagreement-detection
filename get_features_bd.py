import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import string
import nltk
from nltk.tokenize import  word_tokenize
from Feature_Extractor import extract_features

nltk.download('punkt')

# For data pre-processings

contr_dict = {"i\'m" : "i am",
            "won\'t" : "will not",
            "it\'s" : "it is",
            " \'" : " ",
            "\' " : " ",
            "\'s" : "",
            " \'s" : "",
            " s " : " ",
            "\'ll" : "will",
            "\'ve" : "have",
            "n\'t" : "not",
            "\'re" : "are",
            "\'d" : "would",
            "y'all" : "all of you"}

def remove_html(text):
    text = text.replace("< i >", "")
    text = text.replace("< / i >", "")
    return text

def remove_numbers(text):
    no_digits = ""
    for i in text:
        if not i.isdigit():
            no_digits += i
    return no_digits

# remove apostrophes and replaces contractions with full words
def replace_contractions(text, contr_dict=contr_dict):
    text = text.lower()
    for char in ["′","ʼ","`","՚","ʼ","ߴ","ߵ","＇", '“','”',"\'", "’", "‘"]:
      text = text.replace(char, "\'")
    for contr in contr_dict:
      text = text.replace(contr, " "+contr_dict[contr])
    return text

def text_preprocessing(texts):
  new_texts = []
  for text in texts:
      text = remove_html(text)
      text = remove_numbers(text)
      text = replace_contractions(text)
      new_texts.append(text)
  return new_texts

def get_data(input_df):
  print('Loading data')
  X1, X2, Y = [], [], []
  X1 = input_df['title1_en'].tolist()
  X2 = input_df['title2_en'].tolist()
  X1 = text_preprocessing(X1)
  X2 = text_preprocessing(X2)
  assert len(X1) == len(X2)
  return X1, X2

def to_column(data):
  fset = set()
  col = list()
  for row in data:
    for feature in row:
      fset.add(feature)
    col.append(fset)
    fset = set()
  return np.array(col)

# Import ByteDance dataset

le = LabelEncoder()
le2 = LabelEncoder()

# Train set
train_df = pd.read_csv('../original_model/Multitask-NE/emotion_module/train_bd_with_labels_2nd.csv')
train_df['bd_label'] = le.fit_transform(train_df['bd_label']) # agreed = 0, disagreed = 1
train_df['single_new_emo'] = le2.fit_transform(train_df['single_new_emo']) # emotions agree = 0, disagree = 1
train_df['title1_en'], train_df['title2_en'] = get_data(train_df)

# Test set
test_df = pd.read_csv('../original_model/Multitask-NE/emotion_module/test_bd_with_labels.csv')
test_df['bd_label'] = le.transform(test_df['bd_label'])
test_df['single_new_emo'] = le2.transform(test_df['single_new_emo'])
test_df['title1_en'], test_df['title2_en'] = get_data(test_df)

print(train_df.iloc[:10])

# Extract lexicon and other features for train data
print("\nTrain data:")
print("Extracting features of premises...")
pre_aux_features_train_dim, pre_aux_features_train = extract_features(train_df, 'title1_en') # premise
np.save('pre_features_train.npy', pre_aux_features_train)

print("Extracting features of hypotheses...")
hyp_aux_features_train_dim, hyp_aux_features_train = extract_features(train_df, 'title2_en') # hypothesis
np.save('hyp_features_train.npy', hyp_aux_features_train)

print("Premise feature shape: ",pre_aux_features_train.shape)
print("Hypothesis feature shape: ",hyp_aux_features_train.shape)

# Extract lexicon and other features for test data
print("\nTest data:")
print("Extracting features of premises...")
pre_aux_features_test_dim, pre_aux_features_test = extract_features(test_df, 'title1_en') # premise
np.save('pre_features_test.npy', pre_aux_features_test)

print("Extracting features of hypotheses...")
hyp_aux_features_test_dim, hyp_aux_features_test = extract_features(test_df, 'title2_en') # hypothesis
np.save('hyp_features_test.npy', hyp_aux_features_test)

print("Premise feature shape: ",pre_aux_features_test.shape)
print("Hypothesis feature shape: ",hyp_aux_features_test.shape)

print("Done.")

