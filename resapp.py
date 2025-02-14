#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = load_model('face_mask_detection_resnet50.h5')

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the templates folder if using this

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    result = "Wearing a Mask" if prediction < 0.5 else "Not Wearing a Mask"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

