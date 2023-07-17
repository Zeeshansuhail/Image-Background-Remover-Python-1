import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os
import pyrebase

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
from flask import Flask, request
from flask_cors import CORS



app = Flask(__name__)
CORS(app)


# Get The Current Directory
currentDir = os.path.dirname(__file__)

currentPath = os.path.basename(__file__)

config = {
  "apiKey": "AIzaSyBW0mIy7rSWVo1hpXO62PgoueD4VQ2o6Bo",
  "authDomain": "allinonedigitaltools-8563b.firebaseapp.com",
  "databaseURL": "https://allinonedigitaltools-8563b-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "allinonedigitaltools-8563b",
  "storageBucket": "allinonedigitaltools-8563b.appspot.com",
  "messagingSenderId": "957600292234",
  "appId": "1:957600292234:web:c7075ee708e2c5573bbc73",
  "measurementId": "G-K642Y7XP3R",
  "serviceAccount": "serviceAccount.json",
}
# Functions:
# Save Results


def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(d_dir+output_name)
# Remove Background From Image (Generate Mask, and Final Results)
@app.route('/remove', methods=['POST'])
def removeBg():
    imagename =  request.form['downloadname']
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    image_ref = storage.child("original/"+imagename)
    image_ref.download(imagename)
    imagePath = request.form['data']
    inputs_dir = os.path.join(currentDir, 'static/inputs/')
    results_dir = os.path.join(currentDir, 'static/results/')
    masks_dir = os.path.join(currentDir, 'static/masks/')
      
    # convert string of image data to uint8
    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)



    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return '---Empty image---'

    # decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # build a response dict to send back to client
        return "---Empty image---"

    # save image to inputs
    unique_filename = str(uuid.uuid4())
    cv2.imwrite(inputs_dir+unique_filename+'.jpg', img)

    # processing
    image = transform.resize(img, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    save_output(inputs_dir+unique_filename+'.jpg', unique_filename +
                '.png', pred, results_dir, 'image')
    save_output(inputs_dir+unique_filename+'.jpg', unique_filename +
                '.png', pred, masks_dir, 'mask')
    
    print(inputs_dir+unique_filename+'.jpg')
    
    storage.child(unique_filename+'.png').put(results_dir+unique_filename+'.png')
     
    message = {
        "status": "success",
        "output_image": unique_filename + '.png'
    }


    return message


# ------- Load Trained Model --------
print("---Loading Model---")
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
# ------- Load Trained Model --------


print("---Removing Background...")
# ------- Call The removeBg Function --------

# def call():
#     imgPath = "moto.png"
#     print("---Loading Background...")
#     print("---Removing Background...")
#     print(removeBg(imgPath))
#     return "---Success---"

#  # Change this to your image path
#print()

if __name__ == '__main__':
    app.run()
