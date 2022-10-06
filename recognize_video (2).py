import pandas as pd
from datetime import datetime
#from sklearn.preprocessing import LabelEncoder
import sheet
import shutil
import os

def image_recognize(image, all_data):
    
    image = cv2.resize(image, (640, 480))
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage( cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0),swapRB=False, crop=False)
    
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    for i in range(0, detections.shape[2]):
    
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.85:
            # compute the (x, y)-coordinates of the bounding box for the face

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
    
            if fW < 30 or fH < 30:
                continue
                
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            
            preds = recognizer.predict(vec)[0]
            j = np.argmax(preds)

            proba = preds[j]
            name = le.classes_[j]
            if proba >= 0.40:
                text = name

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, text, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

            #! ----------------------------------------------------------------------
            all_data = sheet.save(name, all_data)
            
            #! ----------------------------------------------------------------------


    '''
      #تاريخ اليوم فقط اسم الفولدر      
    #folder_name = str(datetime.today())
    folder_name = datetime.today().strftime('%Y-%m-%d')
    
    
    file_name = datetime.today().strftime('%H:%M:%S')
    
    file_name= str(file_name).replace(':', '.')
    
    #input_path = "E:\\face recognition project1\\output.xlsx"
    #output_path = "E:\\face recognition project1\\results" +'\\'+ str_date +".xlsx" 


                #--------------------------folder name------------------------------------------
    #chice is her folder of now day
    #os.chdir ('E:\\face recognition project1\\results')
    try:
    # Create target Directory
        os.mkdir ( folder_name )
        #os.chdir ('E:\\face recognition project1\\results\\'+folder_name)
    except FileExistsError:
        print("Directory is already exists")
        
    """/////////////////////////////////////////////////////////"""
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        #print("Directory " , folder_name ,  " Created ")
    else:    
        print("Directory " , folder_name ,  " already exists")  

    
    input_path = "E:\\face recognition project1\\"+file_name+".xlsx" #بتخليه يطبع 
    output_path = "E:\\face recognition project1\\results\\"+folder_name+"" 
    '''
    #print(input_path)
    #print(output_path)
    
    #copyfile(input_path,output_path)
    #shutil.copy(input_path, output_path)
    #shutil.copy(input_path, output_path)    
 

    
   
    
   #shutil.move("output.xlsx","results\\"+time+".xlsx")
    #all_data = all_data.save('C:\\Users\madaa\Desktop\face recognition project1\results')
    return all_data, np.array(image)
    pd.DataFrame.from_dict(all_data).to_excel('results\\'+file_name+'.xlsx', index=False)
    #date=datetime.now()
    #str_date= str(date).replace(':', '.')
    #time = str_date.strftime('%c')
    #print(str_date)
    #تاريخ والوقت اسم الملفات      
    #dt = datetime.now()
    #time = dt.strftime('%c')

        
    
    #os.chdir ('E:\\face recognition project1\\results\\'+today)
    #folder_name = datetime.today().strftime('%Y-%m-%d')
    #print(folder_name)    
            #!-------------------file name------------------------------------

        
    

# import libraries
from keras.models import load_model
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from joblib import dump, load

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
#recognizer =pickle.loads(open("output/recognizer.pickle", "rb").read())
recognizer=load_model('recognizer.h5')

#le = pickle.loads(open("output/le.pickle", "rb").read())
le=load('output/le.joblib')

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
all_data = None

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	all_data, frame=image_recognize(frame, all_data)
	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()

folder_name = datetime.today().strftime('%Y-%m-%d')


file_name = datetime.today().strftime('%H:%M:%S')
file_name= str(file_name).replace(':', '.')



#--------------------------folder name------------------------------------------
#chice is her folder of now day
os.chdir ('E:\\face recognition project1\\results')
try:
# Create target Directory
    os.mkdir ( folder_name )
    #os.chdir ('E:\\face recognition project1\\results\\'+folder_name)
except FileExistsError:
    print("Directory is already exists")
    
"""/////////////////////////////////////////////////////////"""
# Create target directory & all intermediate directories if don't exists
os.chdir ('E:\\face recognition project1\\results')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    #print("Directory " , folder_name ,  " Created ")
else:    
    print("Directory " , folder_name ,  " already exists")


os.chdir ('E:\\face recognition project1')
pd.DataFrame.from_dict(all_data).to_excel('results\\'+ folder_name+'\\'+ file_name+'.xlsx', index=False)





#! save to excel sheet
#pd.DataFrame.from_dict(all_data).to_excel("output.xlsx", index=False)  

print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
#cv2.waitkey(fps)
cv2.destroyAllWindows()
vs.stop()

