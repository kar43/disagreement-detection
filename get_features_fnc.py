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

contr_dict = {"I\'m": "I am",
            "won\'t": "will not",
            "\'s" : "", 
            "\'ll":"will",
            "\'ve":"have",
            "n\'t":"not",
            "\'re": "are",
            "\'d": "would",
            "y'all": "all of you",
            "\' " : " ",
            " \'" : " "}

known_typos = {
    "accordin ":"according ",
    "entirel ":"entirely ",
    "electri ":"electric"
}

def encode_char(text):
    text = text.replace(u"\u201d",'"') # replace unicode quotation
    text = text.replace(u"\u201c",'"')
    text = text.replace(u"\u2018","'")
    text = text.replace(u"\u2019","'")
    if u"\u201d" in text or u"\u201c" in text or u"\u2018" in text or u"\u2019" in text:
      print(text)
    encoded_string = text.encode("ascii", "ignore")
    text = encoded_string.decode()
    return text

def correct_typos(text):
    text = text + " "
    for typo in known_typos:
      text = text.replace(typo, " "+known_typos[typo])
    return text

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
    for i in range(len(text)):
      if text[i] == "\'" and (i == 0 or text[i-1] == "."):
        text[i] == ""
    text = text.replace(" \'"," ")
    text = text.replace("\' "," ")
    for contr in contr_dict:
      text = text.replace(contr, " "+contr_dict[contr])
    text = text.replace("\'","")
    return text

def text_preprocessing(texts):
  new_texts = []
  for text in texts:
      text = encode_char(text)
      text = remove_numbers(text)
      text = replace_contractions(text)
      text = correct_typos(text)
      new_texts.append(text)
  return new_texts

def get_data(input_df):
  print('Loading data')
  X1, X2, Y = [], [], []
  X1 = input_df['Body'].tolist()
  X2 = input_df['Headline'].tolist()
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
train_df = pd.read_csv('../original_model/Multitask-NE/emotion_module/train_fnc_with_labels_2nd.csv')
train_df['Stance'] = le.fit_transform(train_df['Stance']) # agreed = 0, disagreed = 1
train_df['single_new_emo'] = le2.fit_transform(train_df['single_new_emo']) # emotions agree = 0, disagree = 1
train_df['Body'], train_df['Headline'] = get_data(train_df)

# Test set
test_df = pd.read_csv('../original_model/Multitask-NE/emotion_module/test_fnc_with_labels.csv')
test_df['Stance'] = le.transform(test_df['Stance'])
test_df['single_new_emo'] = le2.transform(test_df['single_new_emo'])
test_df['Body'], test_df['Headline'] = get_data(test_df)

print(train_df.iloc[:10])

# Extract lexicon and other features for train data
print("\nTrain data:")
print("Extracting features of premises...")
pre_aux_features_train_dim, pre_aux_features_train = extract_features(train_df, 'Body') # premise
np.save('pre_features_train_fnc.npy', pre_aux_features_train)

print("Extracting features of hypotheses...")
hyp_aux_features_train_dim, hyp_aux_features_train = extract_features(train_df, 'Headline') # hypothesis
np.save('hyp_features_train_fnc.npy', hyp_aux_features_train)

print("Premise feature shape: ",pre_aux_features_train.shape)
print("Hypothesis feature shape: ",hyp_aux_features_train.shape)

# Extract lexicon and other features for test data
print("\nTest data:")
print("Extracting features of premises...")
pre_aux_features_test_dim, pre_aux_features_test = extract_features(test_df, 'Body') # premise
np.save('pre_features_test_fnc.npy', pre_aux_features_test)

print("Extracting features of hypotheses...")
hyp_aux_features_test_dim, hyp_aux_features_test = extract_features(test_df, 'Headline') # hypothesis
np.save('hyp_features_test_fnc.npy', hyp_aux_features_test)

print("Premise feature shape: ",pre_aux_features_test.shape)
print("Hypothesis feature shape: ",hyp_aux_features_test.shape)

print("Done.")

