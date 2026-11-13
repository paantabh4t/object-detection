from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#cap = cv2.VideoCapture("video.mp4") for video

model = YOLO("yolov8n.pt")



while True:
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            cvzone.cornerRect(img, (x1,y1,x2,y2))

            conf = math.ceil((box.conf[0]*100)) / 100
            names = model.names
            cls = int(box.cls)
            cvzone.putTextRect(img, f'{names[cls]} {conf}', (x1, y1), scale = 1, thickness = 1)

    
