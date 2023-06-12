
import base64
import datetime
import json
import time
import cv2
import os
import copy
import csv

import numpy as np
from PIL import Image

from pydantic import BaseModel
import pygame

from multiprocessing import Process
from collections import defaultdict
from collections import deque



from fastapi import FastAPI, File, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

from typing import List
# mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# YOLO 
from ultralytics import YOLO 
from super_gradients.training import models
import torch 

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
nas_model = models.get('yolo_nas_l', pretrained_weights="coco").to(DEVICE)

yolo_model = YOLO("models/yolov8l.pt")


from tools import *

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


BASE_PATH  = os.getcwd()
SOUND_FILE = "data/sounds/mixkit-classic-alarm-995.wav"
MODEL_PATH = "tools/keypoint_classifier/keypoint_classifier.tflite"

mp_drawing = mp.solutions.drawing_utils

# Mp object detection - Initialize effi model
base_options = python.BaseOptions(model_asset_path='models/effi/efficientdet_lite2_uint8.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
detector = vision.ObjectDetector.create_from_options(options)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize MediaPipe Face Mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize the motion vars
motion_status = []
motion_duration = 0
distance = 0

no_motion_duration = 3  #  seconds
fps = 10  # Assuming the camera captures 30 frames per second
motion_threshold = 0.5  # Initial value 0.01 - 0.1
keypoint_classifier = None

# Initialize cell phone vars


# Initialize Emotion vars
dominant_emotions = []  # List to store dominant emotions
emotion_data = deque(maxlen=1*60*6)  # 5 fps assuming fps frames per second
shift_start_time = datetime.datetime.now()

# Read labels
with open('tools/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]



'''Fastapi get url with port number'''
@app.get("/")
def read_root(request:Request=None):

    videos={ i:i for i in os.listdir('data/videos/')}
    if os.path.exists('data/initial.txt'): 
        initial_pos = [int(value) for value in open('data/initial.txt', 'r').read().splitlines()]
    else:initial_pos=[10,90,10,90,1,40]
    # Invoke the WSGI application function and return the response

    return templates.TemplateResponse("index.html", {"request": request, "videos": videos, 'pos': initial_pos })

def play_sound(file_path, play):
    if not play:return
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
class LineValuesAndCheckboxes(BaseModel):
    lineValues: List[int]
    checkboxValues: List[bool]
    selectedVideo: str
    start:bool


proces = distance = cap = None
@app.post('/motion-detection-app')
async def motion_detections(data:LineValuesAndCheckboxes, request:Request=None):

    global cap, SOUND_FILE, keypoint_classifier, start, \
        pose, proces, motion_duration, mp, \
        distance,  face_mesh, mp_face_mesh, mp_hands, hands, \
        dominant_emotions, emotion_counts, emotion_data, shift_start_time, emotion_chart

    # Initialize the camera
    no_motion_duration = data.lineValues[4] 
    motion_threshold = data.lineValues[5] / 1000 # 0.01 - 0.1
    smoking_distance_threshold = 0.1
    smoking_detected = False
    cellphone_distance_threshold = 0.2
    mp_cellphone_usage_detected = False
    cellphone_usage_detected = False
    text = ""
    emotion = ""
        
    if not data.start:
        print('reseting....')
        cap.release()
        cap = None
        with open('data/initial.txt', 'w') as f: 
            f.write('\n'.join(map(str, data.lineValues)))
            
        return JSONResponse(status_code=200, content={"detail": "Stopping loop due to condition."})
    
    if cap is None:
        proces = Process(target=play_sound, args=(SOUND_FILE, False, ))
        proces.start()        
        keypoint_classifier = KeyPointClassifier(model_path = MODEL_PATH)        
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        start = 0
        # Emotion init
        dominant_emotions = []
        emotion_counts = defaultdict(int)
        emotion_data = deque(maxlen=1*60*6) 
        emotion_chart =  np.full((200, 200, 4), 255, dtype=np.uint8)  # create an opaque white image
        emotion_chart[:, :, 3] = 0  # make the image fully transparent


    ret, frame = cap.read()
    fps2 = 1/(time.monotonic() - start)
    start=time.monotonic()
    if ret:
        debug_image = copy.deepcopy( cv2.flip(frame.copy(), 1) )
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # MediaPipe pose estimation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False
        # result = pose.process(image)
        hand_results = hands.process(image)
        face_results = face_mesh.process(image)
        image.flags.writeable = True
        
        color = (0,0,255)
        if face_results.multi_face_landmarks:
            
            # Calculate motion

            if motion_status:  # If there are previous keypoints
                #current_keypoints = np.array([[lmk.x, lmk.y] for lmk in result.pose_landmarks.landmark])
                current_keypoints = np.array([[lmk.x, lmk.y] for lmk in face_results.multi_face_landmarks[0].landmark])

                distance = np.linalg.norm(current_keypoints - motion_status[-1])
                
                if distance > motion_threshold:
                    motion_duration = 0
                else:
                    motion_duration += 1 / fps
                
                if motion_duration >= no_motion_duration and not proces.is_alive() :
                    # Trigger alarm
                    print(f"Alarm: No motion detected for {motion_duration} second")
                    proces = Process(target=play_sound, args=(SOUND_FILE, False,))
                    proces.start()
                if proces.is_alive(): color = (255,0,0)
            
            # motion_status.append(np.array([[lmk.x, lmk.y] for lmk in result.pose_landmarks.landmark]))
            motion_status.append(np.array([[lmk.x, lmk.y] for lmk in face_results.multi_face_landmarks[0].landmark]))
            text = f"{motion_duration:.2f}/{round(no_motion_duration)} ({'Dikkat' if proces.is_alive() else f'FPS:{fps2:.0f}'})"
            # Update motion status
        
        # Check if a hand and face are detected in the frame
        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            # Get the landmarks of the index fingertip and the mouth
            index_fingertip = np.array([hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                        hand_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            mouth_upper_lip = np.array([face_results.multi_face_landmarks[0].landmark[13].x,
                                        face_results.multi_face_landmarks[0].landmark[13].y])
            mouth_lower_lip = np.array([face_results.multi_face_landmarks[0].landmark[14].x,
                                        face_results.multi_face_landmarks[0].landmark[14].y])

            # Calculate the distances between the index fingertip and the upper and lower lips
            upper_lip_distance = euclidean_distance(index_fingertip, mouth_upper_lip)
            lower_lip_distance = euclidean_distance(index_fingertip, mouth_lower_lip)
            
            # Check if the index fingertip is close to the mouth
            if upper_lip_distance < smoking_distance_threshold or lower_lip_distance < smoking_distance_threshold:
                smoking_detected = True
            else:
                smoking_detected = False

        # Emotion :
        if face_results.multi_face_landmarks:
            # Get the facial landmarks as a list
            for face_landmarks in face_results.multi_face_landmarks:     
                
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                #emotion classification
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                
                emotion = keypoint_classifier_labels[facial_emotion_id]
                
                emotion_data.append(emotion)
                current_time = datetime.datetime.now()
                
                if (current_time - shift_start_time).seconds >= 1*60:  # every 15 minutes             
                    emotion_chart, dominant_emotions, emotion_data, emotion_counts, current_time, shift_start_time  = \
                        get_emotion_chart(dominant_emotions, emotion_data, emotion_counts, current_time, (200,200))
  
  
        
        image = bind_emotion_chart_on_image(image, emotion_chart)
            

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            mp_cellphone_usage_detected = is_using_cellphone(
                face_results.multi_face_landmarks[0],
                hand_results.multi_hand_landmarks[0], 
                mp_hands,
                cellphone_distance_threshold )
            
        if mp_cellphone_usage_detected:
            # chekc if hand is near the ear is cell phone exists?
            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            image = yolo_detect_n_annotate_imame(image, yolo_model, nas_model, nas=True )
                        
        text = text_update(text, smoking_detected, mp_cellphone_usage_detected, cellphone_usage_detected, emotion)

        yy = 0
        for txt in text.split('\n'):
            yy += 30
            cv2.putText(image, txt, (10,yy), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        is_success, img_buffer = cv2.imencode(".png", image)
        if is_success:
            img_base64 = base64.b64encode(img_buffer).decode("utf-8")
            response_data = {
                "image": img_base64,

            }
            response_json = json.dumps(response_data)
            return Response(response_json, media_type="application/json")
    else:
        cap.release()
        cv2.destroyAllWindows()
        return None
        


""" # Draw hand landmarks
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec, drawing_spec)

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
 """
        