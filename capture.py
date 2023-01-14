# STEP 1:
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
import numpy as np

model_path = 'D:/Documents/Projects/HackNRoll23/backend/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
  
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('something', frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#       break
# cap.release()
# cv2.destroyAllWindows()

while True:
    vid = cv2.VideoCapture(0)
    img, frame = vid.read()
    print(frame.shape)
    print(type(img))
    #mp_image = mp.Image(format=ImageFormat.SRGB, data=np.ndarray(frame))
    with GestureRecognizer.create_from_options(options) as recognizer:
        recognizer.recognize_async(frame)
        result = GestureRecognizerOptions.result_callback
        print(result)