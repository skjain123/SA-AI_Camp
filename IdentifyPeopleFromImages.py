import face_recognition
known_image = face_recognition.load_image_file("./known_people/Barack_Obama.jpeg")
unknown_image = face_recognition.load_image_file("./unknown_pictures/unknown.jpg")

obama_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([obama_encoding], unknown_encoding)
print(results)
