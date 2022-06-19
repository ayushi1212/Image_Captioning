from utils import img_capt, txt_clean, txt_vocab, save_descriptions,extract_features,load_photos
from loader_func import load_clean_descriptions, load_features, create_tokenizer, max_length, data_generator
from model import define_model
from pickle import dump
import os
from pickle import load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='a1', help="Path to Flickr8k.text folder")
parser.add_argument(dest='a2', help="Path to Flicker8k_Dataset (Image folder)")
parser.add_argument(dest='epochs', help="Number of epochs to run")
args = parser.parse_args()

path = os.getcwd()
path_token = args.a1 + '/' + 'Flickr8k.token.txt'
path_train = args.a1 + '/Flickr_8k.trainImages.txt'

#loading the file that contains all data
#map them into descriptions dictionary
descriptions = img_capt(path_token)
print("Length of descriptions =" ,len(descriptions))
#cleaning the descriptions
clean_descriptions = txt_clean(descriptions)
#to build vocabulary
vocabulary = txt_vocab(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving all descriptions in one file
if not os.path.exists("model_res"):
    os.makedirs("model_res")
save_descriptions(clean_descriptions, path + '/model_res/' + "descriptions.txt")

if not os.path.exists(path + '/model_res/' + "features.p"):
   print("..Extracting features...of "+ str(len(os.listdir(args.a2)))+" images")
   features = extract_features(args.a2)
   print("number of features",len(features))
   if not os.path.exists("model_res"):
      os.makedirs("model_res")
   dump(features, open(path + '/model_res/' + "features.p","wb"))
   #to directly load the features from the pickle file.
   features = load(open(path + '/model_res/' + "features.p","rb"))
   print("Features extracted. .........Saving")
else:
  features = load(open(path + '/model_res/' + "features.p","rb"))
  print("Number of features",len(features))



train_imgs = load_photos(path_train)
train_descriptions = load_clean_descriptions(path + '/model_res/' + "descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

tokenizer = create_tokenizer(train_descriptions)
if not os.path.exists("model_res"):
    os.makedirs("model_res")
dump(tokenizer, open('model_res/tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(descriptions)

[a,b],c = next(data_generator(train_descriptions, features, tokenizer, max_length,vocab_size))

print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
model = define_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
# creating a directory named models to save our models
if not os.path.exists("model_res/models"):
    os.makedirs("model_res/models")
for i in range(epochs):
   generator = data_generator(train_descriptions, train_features, tokenizer, max_length,vocab_size)
   model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
   model.save("model_res/models/model_" + str(i) + ".h5")