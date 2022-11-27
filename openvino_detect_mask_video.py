from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


# load our serialized face detector model from disk
faceNet = cv2.dnn.readNet('models/face-detection-adas-0001.xml', 'models/face-detection-adas-0001.bin')
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# load the face mask detector model from disk
maskNet = load_model("/home/pi/Desktop/openvino_facemaskdetector/mask_detector.model")


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
starting_time = time.time()
frame_id = 0
# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	frame_id += 1
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 0.07843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

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
			
			faces = np.array(faces, dtype="float32")
			preds = maskNet.predict(faces, batch_size=32)

	#if len(faces) > 0:
	#	faces = np.array(faces, dtype="float32")
	#	preds = maskNet.predict(faces, batch_size=32)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	elapsed_time = time.time() - starting_time
	fps = frame_id / elapsed_time
	cv2.putText(frame, "FPS=" + str(round(fps,2)), (10,10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
