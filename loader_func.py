from keras.preprocessing.text import Tokenizer
import numpy as np
from utils import load_fp
from pickle import load
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

path = os.getcwd()

def load_photos(filename):
   file = load_fp(filename)
   photos = file.split("\n")[:-1]
   return photos

def load_clean_descriptions(filename, photos):
  #loading clean_descriptions
   file = load_fp(filename)
   descriptions = {}
   i=0
   for line in file.split("\n"):
       words = line.split()
       #print('word',len(words))
       i = i+1
       if len(words)<1 :
           continue
       image, image_caption = words[0], words[1:]
       #print("im",image,"ap",image_caption)
       if image in photos:
          if image not in descriptions:
               descriptions[image] = []
          desc = ' ' + " ".join(image_caption) + ' '
          descriptions[image].append(desc)
       #if i==3:
       #  break
   return descriptions

def load_features(photos):
   #loading all features
   all_features = load(open(path + '/model_res/' + "features.p","rb"))
   #selecting only needed features
   features = {k:all_features[k] for k in photos}
   return features

def dict_to_list(descriptions):
   all_desc = []
   for key in descriptions.keys():
       [all_desc.append(d) for d in descriptions[key]]
   return all_desc

def create_tokenizer(descriptions):
   desc_list = dict_to_list(descriptions)
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(desc_list)
   return tokenizer

def max_length(descriptions):
   desc_list = dict_to_list(descriptions)
   #print('d',desc_list[0])
   return max(len(d.split()) for d in desc_list)

def create_sequences(tokenizer, max_length, desc_list, feature,vocab_size):
   x_1, x_2, y = list(), list(), list()
  # move through each description for the image
   for desc in desc_list:
      # encode the sequence
       seq = tokenizer.texts_to_sequences([desc])[0]
       #print(seq,"seq")
      # divide one sequence into various X,y pairs
       for i in range(1, len(seq)):
          # divide into input and output pair
           in_seq, out_seq = seq[:i], seq[i]
           #print("in_Seq",in_seq,"out_seq",out_seq)
          # pad input sequence
           in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
           #print("pad",in_seq)
          # encode output sequence
           out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
           #print("out_seq",out_seq,list(np.unique(out_seq)),out_seq.shape)
          # store
           x_1.append(feature)
           x_2.append(in_seq)
           y.append(out_seq)
   return np.array(x_1), np.array(x_2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length,vocab_size):
   while 1:
      for key, description_list in descriptions.items():
         # retrieve photo features
         feature = features[key][0]
         inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature,vocab_size)
         yield [[inp_image, inp_seq], op_word]

