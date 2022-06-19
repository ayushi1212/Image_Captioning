import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception
from pickle import load


def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")
    image = image.resize((299, 299))
    image = np.array(image)
    # for 4 channels images, we need to convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
   for word, index in tokenizer.word_index.items():
       if index == integer:
          return word
   return None

def generate_desc(model, tokenizer, photo, max_length):
   in_text = 'start'
   for i in range(max_length):
       sequence = tokenizer.texts_to_sequences([in_text])[0]
       sequence = pad_sequences([sequence], maxlen=max_length)
       pred = model.predict([photo,sequence], verbose=0)
       pred = np.argmax(pred)
       word = word_for_id(pred, tokenizer)
       if word is None:
           break
       in_text += ' ' + word
       if word == 'end':
           break
   return in_text

def pred_params(img_path,tokenfile_path,model_path):
    max_length = 32
    tokenizer = load(open(tokenfile_path, "rb"))
    model = load_model(model_path)
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)
    return model,tokenizer,photo,max_length