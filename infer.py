from loader_func import load_photos
import argparse
import os
from infer_func import pred_params, generate_desc
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(dest='a1', help="Path to Flickr8k.text folder")
parser.add_argument(dest='a2', help="Path to Flicker8k_Dataset (Image folder)")
args = parser.parse_args()

path = os.getcwd()
path_token = args.a1 + '/' + 'Flickr8k.token.txt'
path_test = args.a1 + '/Flickr_8k.testImages.txt'

tokenfile_path = path + '/model_res/tokenizer.p'
model_file_path = path + '/model_res/models'
model_path_len = len(os.listdir(model_file_path))
length = model_path_len-1
model_path = model_file_path + '/' + os.listdir(model_file_path)[length]
test_imgs = load_photos(path_test)

if __name__ == "__main__":
  for i in tqdm(test_imgs):
      img_path =  args.a2 +'/' + i
      model, tokenizer,photo,max_length = pred_params(img_path,tokenfile_path,model_path)
      description = generate_desc(model, tokenizer, photo, max_length)
      print(description)
