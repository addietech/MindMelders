import numpy as np
import cv2
import pickle
import tellopy

face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Resources/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Resources/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")


labels = {"preson_name" : 1}
with open("labels.pickle", 'rb') as f:
og_labels = pickle.load(f)
labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
ret, frame = cap.read() #capture frame by frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
for(x, y, w, h) in faces:

#ROI means region of interest
#print(x, y, w, h)
roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
roi_color = frame[y:y+h, x:x+w] #(ycord_start, ycord_end)

#recognize deep learning model
id_, conf = recognizer.predict(roi_gray)
if conf>=60: ##and conf<=85:
print(id_)
print(labels[id_])

font = cv2.FONT_HERSHEY_SIMPLEX
name = labels[id_]
color = (255, 0, 0)
stroke = 2
cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


img_item = "12.png"
cv2.imwrite(img_item, roi_color)

#prints rectangle
color = (255, 0, 0) #BGR
stroke = 2
end_cord_x = x + w
end_cord_y = y + h
cv2.rectangle(frame, (x, y),(end_cord_x, end_cord_y), color, stroke)
eyes = eye_cascade.detectMultiScale(roi_gray)
for(ex, ey, ew, eh ) in eyes:
cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0), 2)
#sub_items = smile_cascade.detectMultiScale(roi_gray)
#for(ex, ey, ew, eh ) in sub_items:
#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0), 2)


#Display the image on screen
cv2.imshow('frame', frame)
if cv2.waitKey(20) & 0xFF == ord('q'):
break


cap.release()
cv2.destroyAllWindows()