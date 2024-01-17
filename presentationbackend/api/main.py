from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import io
from PIL import Image
import base64
app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("model")

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image.save("received_image.png")

        received_image = np.array(image)
        grayscale_image = cv2.cvtColor(received_image, cv2.COLOR_RGBA2GRAY)

        print("Received Image Shape:", grayscale_image.shape)
        print("Received Image Data:", grayscale_image)

        img = cv2.resize(grayscale_image, (28, 28))
        img = np.invert(np.array([img]))

        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)

        response = {
            'prediction': int(predicted_class_index),
        }

        print(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)