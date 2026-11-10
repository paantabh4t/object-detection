from ultralytics import YOLO
import cv2

model  = YOLO('yolov8m.pt')
results = model("assets/image1.png")
cv2.imshow("result", results[0].plot())
cv2.waitKey(0)