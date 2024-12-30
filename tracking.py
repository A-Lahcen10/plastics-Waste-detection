import cv2
from ultralytics import YOLO
# Load the YOLO model
model = YOLO('best_final.pt')

class_list = model.names

# Open the video file
cap = cv2.VideoCapture('video2.mp4')



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True) 

    
    if results[0].boxes.data is not None:
   
        boxes = results[0].boxes.xyxy
        if results[0].boxes.data is not None:
        
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int()
            else:
                track_ids = []
        class_indices = results[0].boxes.cls.int()
        confidences = results[0].boxes.conf

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
                
            class_name = class_list[int(class_idx)]
            
            cv2.putText(frame, f"ID:  {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow(" Object Tracking", frame)    
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            

