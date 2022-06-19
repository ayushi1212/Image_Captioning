from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
import shutil
import os
from infer_func import pred_params, generate_desc
from tqdm.notebook import tqdm

path = os.getcwd()
path_input = path + '/' + 'api_input'

path = os.getcwd()
tokenfile_path = path + '/model_res/tokenizer.p'
model_file_path = path + '/model_res/models'
model_path_len = len(os.listdir(model_file_path))
length = model_path_len-1
model_path = model_file_path + '/' + os.listdir(model_file_path)[length]

def output(img_fold_path):
    for i in os.listdir(img_fold_path):
        img_path = img_fold_path + '/' + i
        model, tokenizer, photo, max_length = pred_params(img_path, tokenfile_path, model_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        return description

try:
    os.mkdir(path_input)
except OSError:
    print("Creation of the directory %s failed" % path_input)
else:
    print("Successfully created the directory %s " % path_input)

app = FastAPI()

@app.get('/')
def hello_world(name:str):
    return f"Hello {name}!"

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    with open(f'{file.filename}', 'wb') as dst:
        shutil.copyfileobj(file.file, dst)
    f = path + '/' + file.filename
    dest_folder = path_input
    shutil.move(f, dest_folder)
    desc = output(dest_folder)
    if desc is not None:
        shutil.rmtree(path_input)
    return {'Description': desc}



if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1',port = 8080)
