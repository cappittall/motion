
import base64
import json
import cv2
import os
import copy
import csv

import numpy as np
import mediapipe as mp
from pydantic import BaseModel
import pygame
from threading import Thread
from multiprocessing import Process, Queue

from fastapi import FastAPI, File, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from typing import List

import itertools
from emotion.model import KeyPointClassifier

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


BASE_PATH  = os.getcwd()
SOUND_FILE = "data/sounds/mixkit-classic-alarm-995.wav"
MODEL_PATH = "emotion/model/keypoint_classifier/keypoint_classifier.tflite"

mp_drawing = mp.solutions.drawing_utils
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)

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

# Initialize the motion status
motion_status = []
motion_duration = 0
no_motion_duration = 3  #  seconds
fps = 10  # Assuming the camera captures 30 frames per second
motion_threshold = 0.5  # Initial value 0.01 - 0.1
cap = None
keypoint_classifier = None

# Read labels
with open('emotion/model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for inx, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])
        
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def average_y_coordinate(landmarks, indices):
    return sum([landmarks[coord].y for coord in indices]) / len(indices)

def landmarks_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def is_using_cellphone(face_landmarks, hand_landmarks):
    # Define a threshold distance for cell phone usage detection
    cellphone_distance_threshold = 0.15

    # Check if the operator's hand is close to their face using key landmarks
    # You can adjust the landmarks to better suit the detection of cell phone usage
    relevant_landmarks = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, 10),  # Index fingertip to nose
        (mp_hands.HandLandmark.THUMB_TIP, 10),  # Thumb tip to nose
        (mp_hands.HandLandmark.PINKY_MCP, 152)  # Pinky MCP to chin
    ]

    for hand_landmark, face_landmark in relevant_landmarks:
        distance = landmarks_distance(
            hand_landmarks.landmark[hand_landmark],
            face_landmarks.landmark[face_landmark]
        )
        if distance < cellphone_distance_threshold:
            return True

    return False

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


proces = distance = motion_threshold = motion_duration =  None
@app.post('/motion-detection-app')
async def motion_detections(data:LineValuesAndCheckboxes, request:Request=None):

    global cap, SOUND_FILE, keypoint_classifier,  \
        pose, mp_pose, proces, motion_duration,\
        distance,  face_mesh, mp_face_mesh, mp_hands, hands
    # Initialize the camera
    no_motion_duration = data.lineValues[4] 
    motion_threshold = data.lineValues[5] / 1000 # 0.01 - 0.1
    smoking_distance_threshold = 0.1

    smoking_detected = False
    cellphone_usage_detected = False
    
    text = ""
    emotion = ""
    
    
    if not data.start:  
        with open('data/initial.txt', 'w') as f: 
            f.write('\n'.join(map(str, data.lineValues)))
    if cap is None:
        motion_duration = 0
        distance = 0
        proces = Process(target=play_sound, args=(SOUND_FILE, False, ))
        proces.start()        
        keypoint_classifier = KeyPointClassifier(model_path = MODEL_PATH)        
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
       
    ret, frame = cap.read()
    if ret:
        debug_image = copy.deepcopy( cv2.flip(frame.copy(), 1) )
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # MediaPipe pose estimation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False
        result = pose.process(image)
        hand_results = hands.process(image)
        face_results = face_mesh.process(image)
        image.flags.writeable = True
             
        color = (255,0,0)
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if result.pose_landmarks:
            
            # Calculate motion

            if motion_status:  # If there are previous keypoints
                current_keypoints = np.array([[lmk.x, lmk.y] for lmk in result.pose_landmarks.landmark])
                distance = np.linalg.norm(current_keypoints - motion_status[-1])
                
                if distance > motion_threshold:
                    motion_duration = 0
                else:
                    motion_duration += 1 / fps
                
                if motion_duration >= no_motion_duration and not proces.is_alive() :
                    # Trigger alarm
                    print(f"Alarm: No motion detected for {motion_duration} second")
                    proces = Process(target=play_sound, args=(SOUND_FILE, True,))
                    proces.start()
                if proces.is_alive(): color = (0,0,255)
                    # motion_duration = 0
            motion_status.append(np.array([[lmk.x, lmk.y] for lmk in result.pose_landmarks.landmark]))
            text = f"{motion_duration:.2f}/{round(no_motion_duration)} ({'Dikkat' if proces.is_alive() else 'Normal'})"
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

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            cellphone_usage_detected = is_using_cellphone(face_results.multi_face_landmarks[0],
                                                  hand_results.multi_hand_landmarks[0])
            
        
        
        text += " | Sigara: VAR " if smoking_detected else " | Sigara: YOK "
        
        text += " | Tel: VAR." if cellphone_usage_detected else " | Tel: YOK."
        text += f"| Ruhsal D: {emotion}"

        yy = 0
        for txt in text.split('\n'):
            yy += 30
            cv2.putText(annotated_image, txt, (10,yy), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            
        is_success, img_buffer = cv2.imencode(".png", annotated_image)
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
        