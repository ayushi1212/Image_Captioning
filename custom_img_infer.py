import os
from infer_func import pred_params, generate_desc
import argparse
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(dest='path', help="Path to test image folder")
args = parser.parse_args()

path = os.getcwd()
img_fold_path = args.path
tokenfile_path = path + '/model_res/tokenizer.p'
model_file_path = path + '/model_res/models'
model_path_len = len(os.listdir(model_file_path))
length = model_path_len-1
model_path = model_file_path + '/' + os.listdir(model_file_path)[length]

if __name__ == "__main__":
  for i in tqdm(os.listdir(img_fold_path)):
    img_path = img_fold_path + '/' + i
    model, tokenizer,photo,max_length = pred_params(img_path,tokenfile_path,model_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)
