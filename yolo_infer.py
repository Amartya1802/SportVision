from ultralytics import YOLO 
model = YOLO('models/yolo5_last.pt')
model.predict('input/tennisVid.mp4', conf=0.2, save=True)
