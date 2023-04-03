
import base64
import json
import cv2
import os

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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


BASE_PATH  = os.getcwd()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)



# Initialize the motion status
motion_status = []
motion_duration = 0
no_motion_duration = 3  # 3 minutes in seconds
fps = 10  # Assuming the camera captures 30 frames per second
motion_threshold = 0.5  # Initial value 0.01 - 0.1
cap = pose = mp_pose = None




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


proces = distance = motion_threshold = sound_file = motion_duration =  None
@app.post('/motion-detection-app')
async def motion_detections(data:LineValuesAndCheckboxes, request:Request=None):

    global cap, pose, mp_pose, motion_threshold, sound_file, proces, mp, motion_duration, distance, no_motion_duration
    # Initialize the camera
    no_motion_duration = data.lineValues[4] 
    motion_threshold = data.lineValues[5] / 1000 # 0.01 - 0.1
    if not data.start:  
        with open('data/initial.txt', 'w') as f: 
            f.write('\n'.join(map(str, data.lineValues)))
    if cap is None:
        motion_duration =  distance = 0
        proces = Process(target=play_sound, args=(sound_file, False, ))
        proces.start()
        sound_file = 'data/sounds/mixkit-classic-alarm-995.wav'           
        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)
        

    
    ret, frame = cap.read()
    if ret:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # MediaPipe pose estimation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        
        if result.pose_landmarks:
            # Draw pose keypoints on the frame
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
                    print("Alarm: No motion detected for 3 minutes")
                    proces = Process(target=play_sound, args=(sound_file, True,))
                    proces.start()
                    # motion_duration = 0
        text = f"Hareketsizlik: {motion_duration:.3f}/{round(no_motion_duration)} ({'Dikkat' if proces.is_alive() else 'Normal'}) H. Fark :{distance:.3f} Fps: {fps}, Hassasiyet:{motion_threshold *1000:.0f}"
        # Update motion status
        motion_status.append(np.array([[lmk.x, lmk.y] for lmk in result.pose_landmarks.landmark]))
        cv2.putText(annotated_image, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
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