# Ms-Project
**Part One**
import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # loading initial weights and model
        protoTxtPath = "face_detector/deploy.prototxt"
        caffeModelPath = "face_detector/weights.caffemodel"
        self.model = cv2.dnn.readNetFromCaffe(protoTxtPath, caffeModelPath)  # initialising(reads the network model) detector with weights

    def process(self, image):
        (h, w) = image.shape[:2] # accessing the image.shape tuple and taking elements

        # preparing the input image for the model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # injecting the blob image to the model 
        self.model.setInput(blob)

        # forward propagation of the image to the model to extract detected faces
        detectedFaces = self.model.forward()

        # checking the confidence of the predictions
        faces = []
        for i in range(0, detectedFaces.shape[2]):
            box = detectedFaces[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidence = detectedFaces[0, 0, i, 2]
            
            if confidence >= 0.6:
                print("confidence",confidence)
                if len(faces) == 0:
                    faces = np.expand_dims(box, axis=0)
                else:
                    faces = np.vstack((faces, box))
        return faces


faceDetector = FaceDetector()


def detectFace(image):
    faceBox = faceDetector.process(image)
    try:
        faces = [(box[1], box[2], box[3], box[0]) for box in faceBox.astype("int")] # getting the start and end points of the detected faces
    except Exception as e:
        faces = list()
    return faces


img = cv2.imread("C:/Users/Zeba Naaz/Desktop/33.jpeg")
print("points",detectFace(img))

**Part two**
import logging
import os
#import traceback

import cv2
import face_recognition
import numpy as np

from PartOne import FaceDetector

faceDetector = FaceDetector()
logger = logging.getLogger(__name__)
threshold = 0.7
path_images = "images"


def detectFace(image):
    faceBox = faceDetector.process(image)
    try:
        faces = [(box[1], box[2], box[3], box[0]) for box in faceBox.astype("int")]
    except Exception as e:
        logger.info(e)
        faces = list()
    return faces


def getFeatures(img, box):
    features = face_recognition.face_encodings(img, box)
    return features


def compareFaces(face_encodings, db_features, db_names):
    matchedPerson = list()
    names = db_names
    features = db_features

    for face_encoding in face_encodings:
        try:
            dist = face_recognition.face_distance(features, face_encoding)
        except Exception as e:
            logger.info(e)
            dist = face_recognition.face_distance([features], face_encoding)

        index = np.argmin(dist)

        if dist[index] <= threshold:
            matchedPerson += [names[index]]
        else:
            matchedPerson += ["unknown"]
    return matchedPerson


def loadImages():
    list_images = os.listdir(path_images)

    # filter files that are not images
    list_images = [File for File in list_images if File.endswith(('.jpg', '.jpeg', 'JPEG', '.png'))]

    # initialize variables
    names = list()
    features = list()

    # image loading
    for file_name in list_images:
        im = cv2.imread(path_images + os.sep + file_name)

        # get the characteristics of the face
        box_face = detectFace(im)
        feat = getFeatures(im, box_face)

        if len(feat) == 1:
            # insert the new characteristics in the database
            new_name = file_name.split(".")[0]
            if new_name:
                names.append(new_name)
                if len(features) == 0:
                    features = np.frombuffer(feat[0], dtype=np.float64)
                else:
                    features = np.vstack((features, np.frombuffer(feat[0], dtype=np.float64)))
    return names, features
    
    
**** Part Three**

import traceback

import cv2
import numpy as np

import PartTwo as FR


class OccludedDetection:
    def __init__(self):
        self.names, self.features = FR.loadImages()

    def recognize(self, img):
        try:
            # detecting face
            box_faces = FR.detectFace(img)

            # if no face is detected
            if not box_faces:
                result = {'status': 'No Face Detected', 'faces': [], 'names': []}
                return result
            else:
                if not self.names:
                    result = {'status': 'Unknown Face Detected', 'faces': box_faces,
                              'names': ['unknown'] * len(box_faces)}
                    return result
                else:
                    actual_features = FR.getFeatures(img, box_faces)
                    # compare actual features with trained images
                    match_names = FR.compareFaces(actual_features, self.features, self.names)
                    result = {'status': 'Face Detected', 'faces': box_faces, 'names': match_names}
                    return result

        except Exception as ex:
            error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            result = {'status': 'error: ' + str(error), 'faces': [], 'names': []}
            return result


def bounding_box(image, box, match_name=[]):
    for i in np.arange(len(box)):
        y0, x1, y1, x0 = box[i]
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3)
        if match_name:
            cv2.putText(image, match_name[i], (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

**Web App**

import cv2
import imutils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from PartThree import OccludedDetection, bounding_box

# initialising recognizer
recognizer = OccludedDetection()
app = Flask(__name__)


def prediction(image):
    frame = cv2.imread(image)
    frame = imutils.resize(frame, width=720)#resizing the picture
    res = recognizer.recognize(frame)
    frame = bounding_box(frame, res["faces"], res["names"])
    cv2.imwrite("{}.jpg".format(res["names"][0]), frame)
    return res["names"]
@app.route("/", methods=["POST", "GET"])


def predict():
    if request.method == 'GET':
        return render_template('index.html')#templates
    elif request.method == "POST":
        image = request.files.get('imgup')
        image.save('./' + secure_filename(image.filename))
        names = prediction(image.filename)
        kwargs = {'name': names[0]}
        return render_template('index2.html', **kwargs)
if __name__ == '__main__':
    app.run()

**Main**
import time

import cv2
import imutils

from PartThree import OccludedDetection, bounding_box

# initialising recognizer
recognizer = OccludedDetection()


def main(inputType):
    if inputType == "webcam":
        cam = cv2.VideoCapture(0)
        while True:
            starTime = time.time()
            _, frame = cam.read()
            frame = imutils.resize(frame, width=720)
            res = recognizer.recognize(frame)
            print(res)

            frame = bounding_box(frame, res["faces"], res["names"])
            endTime = time.time() - starTime
            FPS = 1 / endTime
            cv2.putText(frame, f"FPS: {round(FPS, 3)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("face_recognition", frame)
            cv2.imwrite("test.jpg", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif inputType == "image":
        path_im = "./test_image.jpeg"
        frame = cv2.imread(path_im)
        frame = imutils.resize(frame, width=720)
        res = recognizer.recognize(frame)
        print(res)
        frame = bounding_box(frame, res["faces"], res["names"])
        cv2.imshow("face_recognition", frame)
        cv2.imwrite("{}.jpg".format(res["names"][0]), frame)
        cv2.waitKey(0)


if __name__ == "__main__":
    main(inputType='image')
    # main(inputType="webcam")



