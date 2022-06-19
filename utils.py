import numpy as np
from PIL import Image
import os
import string
from pickle import dump
from pickle import load
from tensorflow.keras.applications.xception import Xception #to get pre-trained model Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer #for text tokenization
from keras.preprocessing.text import Tokenizer
from tqdm.notebook import tqdm  #to check loop progress

#To load text file with image name and descriptions
def load_fp(filename):
  # Open file to read
   file = open(filename, 'r')
   text = file.read()
   file.close()
   return text

#One image contains five caption, therefore file is represented with names of
# those five images along with their captions. This function make sure file name is not repeated,
# while grouping all the captions of same image in a list.
def img_capt(filename):
   file = load_fp(filename)
   captions = file.split('\n')
   descriptions ={}
   for caption in captions[:-1]:
       img, caption = caption.split('\t')
       if img[:-2] not in descriptions:
           descriptions[img[:-2]] = [ caption ]
       else:
           descriptions[img[:-2]].append(caption)
   return descriptions

#cleans caption
def txt_clean(captions):

   table = str.maketrans('','',string.punctuation)

   for img,caps in captions.items():
      for i,img_caption in enumerate(caps):
           img_caption.replace("-"," ")
           descp = img_caption.split()

          #uppercase to lowercase
           descp = [wrd.lower() for wrd in descp]

          #remove punctuation from each token
           descp = [wrd.translate(table) for wrd in descp]

          #remove hanging 's and a
           descp = [wrd for wrd in descp if(len(wrd)>1)]

          #remove words containing numbers with them
           descp = [wrd for wrd in descp if(wrd.isalpha())]

          #converting back to string
           img_caption = ' '.join(descp)
           captions[img][i]= img_caption
   return captions

#creates vocab disctionary
def txt_vocab(descriptions):
   # To build vocab of all unique words
   vocab = set()
   for key in descriptions.keys():
       [vocab.update(d.split()) for d in descriptions[key]]
   return vocab

def save_descriptions(descriptions, filename):
   lines = list()
   for key, desc_list in descriptions.items():
       for desc in desc_list:
           lines.append(key + '\t' + desc )
   data = "\n".join(lines)
   file = open(filename,"w")
   file.write(data)
   file.close()

def extract_features(directory):
       model = Xception(include_top=False, pooling='avg')
       features = {}
       for pic in tqdm(os.listdir(directory)):
           file = directory + "/" + pic
           image = Image.open(file)
           image = image.resize((299, 299))
           image = np.expand_dims(image, axis=0)
           image = image / 127.5
           image = image - 1.0
           feature = model.predict(image)
           features[pic] = feature
       return features

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
   all_features = load(open("/content/features.p","rb"))
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