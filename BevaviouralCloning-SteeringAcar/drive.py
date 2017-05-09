import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
ch, row, col = 3, 30, 100  # Trimmed image format
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array =image_array[30:130, 0:320,:]
    image_array  = cv2.resize(image_array,(64,64), interpolation=cv2.INTER_AREA)  
    transformed_image_array = image_array[None, :,:, :]
 
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if(speed>15.0):
       throttle=0.0
    else:
       throttle = max(0.15,  -0.1/0.05 * abs(steering_angle) + 0.45)
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    # The current steering angle of the car
    #steering_angle = environ["steering_angle"]
    # The current throttle of the car
    #throttle = environ["throttle"]
    # The current speed of the car
    #speed = environ["speed"]
    # The current image from the center camera of the car
    # The driving model currently just outputs a constant throttle. Feel free to edit this.

    send_control(0.0 , 0.2)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

from keras.models import model_from_json
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    #model = model_from_json(json_string)

   # model = load_model(args.model)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5") 
    print("Loaded model from disk")
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
