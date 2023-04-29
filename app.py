import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, abort
import pickle
import imghdr
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from PIL import Image
from PIL import Image

from torchvision import transforms
from torchvision.transforms import ToTensor


app = Flask(__name__)

# the random forest model
rf_model = pickle.load(open('rf_absorptance_model_simplified.pkl','rb'))
# the convnext deep learning model
dl_model  = convnext_tiny(weights=None)
dl_model.classifier[2] = nn.Linear(768, 1)
checkpoint = torch.load("dl_absorptance_weights", map_location=torch.device('cpu'))
dl_model.load_state_dict(checkpoint['model_state_dict'])
device = 'cpu'
dl_model.to(device)

test_preprocess =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/rf_predict',methods = ['POST'])
def rf_predict():
    user_inputs = request.form.values()
    features = [x for x in user_inputs]
    user_inputs = validate_inputs(features)
    # organize features
    int_features = [float(x) for x in features]
    depth, width_half, area = int_features[0], int_features[1], int_features[2]
    aspect_ratio = depth/width_half if width_half != 0 else 0
    int_features = [depth, aspect_ratio, area]

    final_features = [np.array(int_features)]
    prediction = rf_model.predict(final_features)

    return render_template('home.html', prediction_text="RF model predicted absorptance: {}%".format(round(prediction[0],2)))

def validate_inputs(lst):
    '''
    Validate the lst only contain numbers.
    '''
    for element in lst:
        if not isinstance(element, (int, float)) and not (isinstance(element, str) and element.isdigit()):
            abort(400, 'Input must be numbers')


def validate_image(image):
    # Check that the file extension is .tif
    if not image.filename.endswith('.tif'):
        abort(400, 'File must be in .tif format')
    # Check that the file is a valid TIFF image
    if imghdr.what(image) != 'tiff':
        abort(400, 'File is not a valid TIFF image')
    # Check that the image is a gray scale image
    image = Image.open(image)
    channels = image.getbands()
    if len(channels) != 1: 
        abort(400, 'File has to a gray scale tif image')

def preprocess_image(image):
    image = np.array(Image.open(image))
    image = torch.tensor(image,dtype=torch.float64).unsqueeze_(0)
    image = image.repeat(3, 1, 1)
    return image

@app.route('/dl_predict',methods = ['POST'])
def dl_predict():
    image = request.files['image']
    validate_image(image)
    # preprocess iamge
    image = preprocess_image(image)
    image = image.unsqueeze(0)
    image = image.float().to(device)
    # model prediction
    output = dl_model(image) 
    output= float(output.squeeze(1).detach().numpy()[0])
    #
    return render_template('home.html', prediction_text="DL model predicted absorptance: {}%".format(round(output,2)))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)