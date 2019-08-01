import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import os

path = "./img/Sunil_Jain.jpg"
print(path)
def process_image(path):
    print('began process...')
    img = Image.open(path)
    w, h = img.size
    print(w, h)
    #resizing img
    ratio = (320 / float(img.size[0]))
    print(ratio)
    hsize = int((float(h) / float(ratio)))
    wsize = int((float(w) / float(ratio)))
    img = img.resize((wsize, hsize), Image.ANTIALIAS)
    image_title = "./img/resized_image.jpg"
    os.remove(path)
    img.save(image_title)
    print("person has been loaded & resized")
    img = Image.open(image_title)
    new_image = face_recognition.load_image_file(image_title)
    new_face_encoding = face_recognition.face_encodings(new_image)[0]
    print("person learned")
    return (img,new_face_encoding)

process_image(path)
print("done processing...")
