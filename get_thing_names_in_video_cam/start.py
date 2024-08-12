import cv2

# Replace these paths with your actual file locations
classnames = []  # Array to store class names
classfile = 'files/thing.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Load the model
net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Open the video capture
cap = cv2.VideoCapture(0)  # Change 0 to your video file path if needed

# Loop through each video frame
while True:
    ret, img = cap.read()

    # Check if frame reading failed
    if not ret:
        print("Error reading frame from video")
        break

    # Perform object detection on the frame
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Your processing logic here
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, classnames[classId - 1], (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
    else:
        print("No objects detected in this frame")
  
  

    # Display the processed frame
    cv2.imshow('Rakwan', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
