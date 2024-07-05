from ultralytics import YOLO 
model = YOLO('yolov8x.pt')
model.predict('input/tennisVid.mp4', conf=0.2, save=True)
