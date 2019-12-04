import urllib
from urllib.request import urlopen
import discord
from discord.ext import commands
import cv2
import csv
import numpy as np


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

print('Loading models...')
# Pre trained models obtained from Gil Levi and Tal Hassner (age, gender)
# and from VGGFace2 (identity)
age_net = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt',
    'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt',
    'gender_net.caffemodel')
identity_net = cv2.dnn.readNetFromCaffe(
    'senet50_256.prototxt',
    'senet50_256.caffemodel')

# mean values for age and gender
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# mean values for identity
MODEL_MEAN_VALUES_2 = (91.4953, 103.8827, 131.0912)

# Trained Haar classifier cascade used to identify human faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# read the image
#img = cv2.imread("C:\\Users\\MB\\Desktop\\bill3.jpg")
img = url_to_image('https://www.biography.com/.image/t_share/MTE4MDAzNDEwNzg5ODI4MTEw/barack-obama-12782369-1-402.jpg')

# set font to be used for classification details
font = cv2.QT_FONT_NORMAL

# labels
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

Class_ID = []
Name = []
Sample_Num = []
Flag = []
Gender = []
with open('identity_meta.csv', encoding="utf8") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        Class_ID.append(row[0])
        Name.append(row[1])
        Sample_Num.append(row[2])
        Flag.append(row[3])
        Gender.append(row[4])

idw = Name
identity_list = idw

# convert and read the image as a gray-scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect the number of faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)
print("Number of faces detected: {}\n".format(len(faces)))


for x, y, w, h in faces:
    # create rectangular box around each face detected
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_img = img[y:y + h, x:x + w].copy()

    # pass the mean values to the image to pre-process the image so it can be passed to the model for prediction
    blobAgeGender = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    blobIdentity = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES_2, swapRB=False)

    # Predict age
    age_net.setInput(blobAgeGender)
    age_predictions = age_net.forward()
    print("Age Prediction: {}".format(age_predictions))
    age = age_list[age_predictions[0].argmax()]
    print("Age Range: {}".format(age))

    # Predict gender
    gender_net.setInput(blobAgeGender)
    gender_predictions = gender_net.forward()
    print("Gender Prediction: {}".format(gender_predictions))
    gender = gender_list[gender_predictions[0].argmax()]
    print("Gender: {}".format(gender))

    # Predict identity/look-alike
    identity_net.setInput(blobIdentity)
    identity_predictions = identity_net.forward()
    identity = identity_list[identity_predictions[0].argmax()]
    print("Identity or Celebrity Look Alike: {}".format(identity))

    # write labels onto image
    label = "Age Range: {}, Gender: {}, Identity: {}".format(age, gender, identity)
    cv2.putText(img, label, (x, y - 20), font, 1.2, (227, 227), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
