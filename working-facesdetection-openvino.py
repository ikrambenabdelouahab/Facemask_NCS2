from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from threading import Thread


faceNet = cv2.dnn.readNet('models/face-detection-adas-0001.xml', 'models/face-detection-adas-0001.bin')
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)



def ReadStreamAndProcessIt(streamReader):
    frame_id = 0
    starting_time = time.time()
    soundThread = None
    while True:
        frame = streamReader.read()
        frame = imutils.resize(frame, width=460)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        frame_id += 1
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        #preds = []
        

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
                facesAsNumpy  = np.array(faces, dtype="float32")
                #preds = maskNet.predict(facesAsNumpy , batch_size=32)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0) , 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS=" + str(round(fps, 2)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            soundThread = None
            break

ReadStreamAndProcessIt(vs)
