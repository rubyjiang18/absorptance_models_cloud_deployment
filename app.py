import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, abort
import pickle
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from utils import validate_image, validate_inputs, preprocess_image, test_preprocess


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


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/rf_predict',methods = ['POST'])
def rf_predict():
    # input features are in the order of depth (um), width_at_half_depth (um), area (um^2)
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


def main():
    app.run(host='0.0.0.0', port=8081, debug=False)
    
if __name__ == '__main__':
    main()