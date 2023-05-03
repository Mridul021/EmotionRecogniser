from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Dropout, BatchNormalization
from keras import models, layers
from keras.utils import img_to_array


# # REDEFINE MODEL ARCHITECTURE

input_shape = (72, 72, 1)
num_classes = 7
model = models.Sequential()

# First
model.add(layers.Conv2D(32,kernel_size=(3,3),activation = 'relu', padding='same', input_shape = input_shape))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Second
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Third
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.30))

# Fully connected
model.add(layers.Flatten())

# Input layer includes 1024 nodes
model.add(layers.Dense(512, activation='relu'))

# Hidden layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

new_model = keras.models.load_model('FER_model.h5')
face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global filename
    img = request.files['image']
    img.save('static\{}.jpg'.format(img.filename)) #Save image uploaded on web in static folder.
    filename = img.filename
    img_arr = cv2.imread("static\{}.jpg".format(img.filename))
    img_show = np.copy(img_arr) # Clone to show on the web
    img_show_gray = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img_show_gray) # Detect face
    for (x,y,w,h) in faces:
        cv2.rectangle(img_show,(x,y),(x+w,y+h),(0,255,0),5)
        img_arr = img_arr[y:y+h,x:x+w]
        break
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY) # Convert to gray img
    cv2.imwrite('static/{}.jpg'.format(img.filename), img_show)
    img_arr = cv2.resize(img_arr,(72,72),interpolation = cv2.INTER_AREA) # Reshape image to feed to CNN
    img_arr = img_arr / 255.0
    img_arr = img_to_array(img_arr)
    img_arr = np.expand_dims(img_arr,axis=0)
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    prediction = new_model.predict(img_arr)[0]
    label=emotion_labels[prediction.argmax()]
    accuracy = np.round(prediction[prediction.argmax()],2)
    return render_template('prediction.html', data=[label,accuracy])

@app.route('/load_img')
def load_img():
    global filename
    return send_from_directory('static', "{}.jpg".format(filename))


if __name__ == '__main__':
    app.run(debug=True)