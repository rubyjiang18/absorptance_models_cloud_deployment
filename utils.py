from flask import abort
import imghdr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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
    width, height = image.size
    if len(channels) != 1: 
        abort(400, 'File has to a gray scale tif image')
    if width > 350 or height > 350:
        abort(400, 'Image has to be 350x350 pixels or smaller')

def preprocess_image(image):
    image = Image.open(image)
    width, height = image.size
    image_array = np.array(image)
    # if less than dimension 350x250, pad the image using mean
    dimension = 350
    if width < dimension or height < dimension :
        mean_value = np.mean(image_array)
        d_height = dimension - height
        d_width = dimension - width
        top, bottom, left, right = 0, 0, 0, 0
        if d_height > 0:
            top = d_height // 2
            bottom = d_height - top
        if d_width > 0:
            left = d_width // 2
            right = d_width - left
        image_array = np.pad(image_array, ((top, bottom), (left, right)), mode='constant', constant_values=mean_value)
    # convert to tensor
    image = torch.tensor(image_array,dtype=torch.float64).unsqueeze_(0)
    image = image.repeat(3, 1, 1)
    return image

test_preprocess =  transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
