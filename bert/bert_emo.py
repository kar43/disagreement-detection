# All general imports
import torch
import transformers as ppb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer 

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Bidirectional, GlobalAveragePooling1D, GRU, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.layers import LSTM, GRU, Conv1D, MaxPool1D, Activation, Add

from keras.models import Model, Sequential

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K

from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import io, os, gc

if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("cuda")
else:  
  dev = "cpu" 
print(dev) 
device = torch.device(dev)  

# Data Loader for embeddings
def get_embeddings(input_id, attention_mask, model, name):
	store = list()
	extra = list()
	length = input_id.shape[0]
	interval_size = 100
	start = 0
	while(start<length):
		print(start)
		if (start+100) < length:
			des = start+100
			pre_batch_in = input_id[start:des,:]
			pre_batch_at = attention_mask[start:des,:]
			with torch.no_grad():
				last_hidden_states = model(pre_batch_in, attention_mask=pre_batch_at)
			store.append(last_hidden_states[0][:,0,:].cpu().numpy())
		else:
			des = length+1
			pre_batch_in = input_id[start:des,:]
			pre_batch_at = attention_mask[start:des,:]
			with torch.no_grad():
				last_hidden_states = model(pre_batch_in, attention_mask=pre_batch_at)
			extra.append(last_hidden_states[0][:,0,:].cpu().numpy())
		start+=100

	store_np = np.stack(store)
	store_np = store_np.reshape(store_np.shape[0]*store_np.shape[1],768)
	extra_np = np.stack(extra)
	extra_np = extra_np.reshape(extra_np.shape[0]*extra_np.shape[1],768)
	print('store',store_np.shape)
	print('extra',extra_np.shape)
	final_np = np.concatenate([store_np, extra_np], axis=0)
	np.save(name, final_np)
	print(final_np.shape)
	return final_np 

def prepare_bert_embeddings(input_df, save):
	# For DistilBERT:
	model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

	## Want BERT instead of distilBERT? Uncomment the following line:
	#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

	# Load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights)
	model.to(torch.device(device))

	# Applying tokenization
	tokenized = input_df["text_clean"].apply((lambda x: tokenizer.encode(x[:510], add_special_tokens=True)))

	max_len = 0
	for i in tokenized.values:
		if len(i) > max_len:
			max_len = len(i)

	padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

	# Attention masks
	attention_mask = np.where(padded != 0, 1, 0)
	attention_mask.shape

	# Creating input ids
	input_ids = torch.tensor(padded).to(device)  
	attention_mask = torch.tensor(attention_mask).to(device)

	# Getting the embeddings
	bert = get_embeddings(input_ids, attention_mask, model, save)


# Importing the datasets

import json

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data

datapath_train = "../GoEmotions_Dataset/goemotions_train.csv"
datapath_test = "../GoEmotions_Dataset/goemotions_test.csv"

train_df = pd.read_csv(datapath_train)
test_df = pd.read_csv(datapath_test)

prepare_bert_embeddings(train_df, "goemotions_bert_train")
prepare_bert_embeddings(test_df, "goemotions_bert_test")