# FaceRecogntion-attandance-system
The core definition of the system can be summed up in identifying the students by their faces to mark up as attendant. Our system identifies multiple faces in real time by video capturing the lecture hall. First step after capturing live video, face detection is done face recognition happens then the system searches for every face match in the database and the result at the end is an excel sheet with names of these students and some other data the instructor needs to know. We implemented face detection part by OpenCv and deep learning the next step was implemented to extract embbeddings from the images using pre-trained OpenFace model.
# OpenFace model
you can download openface model from this link https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7 
# -El-banhawy-FaceRecogntion-attandance-system
