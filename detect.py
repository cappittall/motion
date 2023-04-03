import base64
from datetime import datetime
from io import BytesIO
from typing import List
import warnings
import os
import time
from csv import writer
import json

# from periphery import GPIO
# import serial

from multiprocessing import Process, Queue
from threading import Thread
from queue import Empty

from fastapi import FastAPI, File, Request, Response

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse



""" from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc 
from google.protobuf.wrappers_pb2 import Int64Value
from tensorflow_serving.apis.model_pb2 import ModelSpec
 """
from pydantic import BaseModel
import numpy as np

import cv2
import time
import re

BASE_PATH  = os.getcwd()
## YOLO 
# from ultralytics import YOLO

# MediaPipe
""" import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles """

#roboflow 
""" from roboflow import Roboflow
rf = Roboflow(api_key="uaiFsEAxwBO5dsU12XBh")
project = rf.workspace().project("ppe-detect-person-in-restricted-zone") """

# tensoflow lite
# import tflite_runtime.interpreter as tflite
import tensorflow as tf




# Detectron2
# Some basic setup:
# Setup detectron2 logger
""" import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() """

# import some common detectron2 utilities
""" from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog """

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

warnings.filterwarnings('ignore')
DETECT_MODEL_PATH = BASE_PATH + '/models/detect' 
### Previous constants  ###
SERVER="localhost:8500"

# Initialize the process and queues
detect_process = None
result_queue = Queue()

VIDEO_PATH = 'data/videos/gun1-light.mp4'
model_paths = ["models/tflite/tflite100/effi2ppe100.tflite", 
               "models/tflite/tflite250/effi2ppe250.tflite"]

model_path = model_paths[1]
processes = []
REDETECT=True
DETECTION_THRESHOLD = 0.4
distance = None
classes = [""]
scores=[]
frame_count = 0
alert = False
trackerName = 'csrt'
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}
trackers = []
# log writing permission
write_control = True
det_line_pre=""
try: 
    det_lines = open(f'data/logs/log{time.strftime("%d%m%Y")}.csv', 'r' ).read().splitlines()
except:
    det_lines = []

""" Sensor and ALARM connections setups
## 40 PIN OUT CONTROL
ALARM = GPIO("/dev/gpiochip2", 13, "out")  # pin 37
## Configure the serial connection
uart_port = "/dev/ttymxc0"  # Use "/dev/ttymxc0" for UART1 on Coral Dev Board
baud_rate = 9600
# Initialize the serial connection
ser = serial.Serial(uart_port, baud_rate, timeout=1) 

"""



# Normalize and batchify the image
def pre_process_image(image_np, type='normalized', input_size=(256,256)):
    if type=='normalized': 
        image = cv2.resize(image_np, input_size)
        image = np.expand_dims(image/255.0, axis=0).astype(np.float32)
    else:    
        image = np.expand_dims(image_np, axis=0).astype(np.uint8)
    return image
class LineValuesAndCheckboxes(BaseModel):
    lineValues: List[int]
    checkboxValues: List[bool]
    selectedVideo: str
    start:bool

cap=predictor=cfg=interpreter=None

'''Fastapi get url with port number'''
@app.get("/")
def read_root(request:Request=None):
    try: 
        cap.release()
        cap = None
    except: pass

    videos={ i:i for i in os.listdir('data/videos/')}
    if os.path.exists('data/initial.txt'): 
        initial_pos = [int(value) for value in open('data/initial.txt', 'r').read().splitlines()]
    else:initial_pos=[10,90,10,90,40]
    # Invoke the WSGI application function and return the response
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos, 'pos': initial_pos })

# check is any detection in restricted area
def check_people_in_restricted_area(cw, ch, data, width, height):
    # Data :  lineValues=[10, 90, 10, 90] checkboxValues=[True, True, True, True]
    _left, _right, _top, _bottom, _  = data.checkboxValues
    # percent of lines from left,right, top ,bottom
    per_left, per_right, per_top, per_bottom, _  = [i/100 for i in data.lineValues]
    val_left = int(width * per_left) if _left else 0
    val_right = int(width * per_right) if _right else width
    val_top = int(height * per_top) if _top else 0 
    val_bottom = int(height * per_bottom) if _bottom else height
    if cw > val_left and cw < val_right and ch > val_top and ch < val_bottom:
        return True
    return False

## Tensorflow lite prediction side  
def preprocess_input(image, input_size):
    resized_image = cv2.resize(image, input_size )
    normalized_image = resized_image.astype(np.uint8) #/ 255.0
    input_data = np.expand_dims(normalized_image, axis=0)
    return input_data



def get_input_size(interpreter):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1:3]  # Assuming the input shape is in the format [batch, height, width, channels]
    return input_size

def extract_detected_objects(interpreter, min_score_thresh = 0.5 ):
    output_details = interpreter.get_output_details()
    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    num_detections = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])    
    detected_objects_list = []
    for i in range(int(num_detections[0])):
        if scores[0][i] >= min_score_thresh:
            detected_obj = {
                'box': boxes[0][i].tolist(),
                'class': int(classes[0][i]),
                'score': scores[0][i]
            }
            detected_objects_list.append(detected_obj)
            
    return detected_objects_list

@app.post('/isg-tflite-model')
async def isg_tflite_model_test(data:LineValuesAndCheckboxes, request:Request=None):

    DETECTION_THRESHOLD = data.lineValues[4] / 100
    global cap, interpreter, VIDEO_PATH, model_path
    if not data.start:
        try:
            VIDEO_PATH = "data/videos/" + data.selectedVideo
            cap.release()
            cap = None
        except Exception as e :print('Error', e)
        return None

    now = datetime.now()
    fileext = now.strftime("%H%M%S") + now.strftime("%f")[:3]
    out_write_path = re.sub('effi2ppe\d+\.tflite', f'out/{fileext}.jpg', model_path)
    in_write_path = re.sub('effi2ppe\d+\.tflite', f'in/{fileext}.jpg', model_path)
    
    wrt = False
    if cap is None:
        with open('data/initial.txt', 'w') as f: f.write('\n'.join(map(str, data.lineValues)))

        cap = cv2.VideoCapture(VIDEO_PATH)
        #roboflow 
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
    success, im = cap.read()
    if success:
        frame = cv2.resize(im, (960, 640))
        input_size = get_input_size(interpreter)
        input_data = preprocess_input(frame.copy(), input_size=input_size)
        
        ## Measure distance
        #distance = read_distance(ser)
        if distance is not None:
            distance_text = f"Distance: {distance} cm"
        else:
            distance_text = "Distance: N/A"
        
        tt= time.time()        
        run_prediction(interpreter, input_data)
        detected_objects = extract_detected_objects(interpreter, min_score_thresh=DETECTION_THRESHOLD)
        print(f'Detection time {time.time()-tt:.3f}' )
        boxes = [obj['box'] for obj in detected_objects]
        height, width, _ = frame.shape
        for obj in detected_objects:
            class_idx = obj['class']
            score = obj['score']

            # Convert coordinates from relative to absolute
            y_min, x_min, y_max, x_max = obj['box']
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
            wrt = True
            ww , hh = (x_max - x_min), (y_max - y_min)
            cw, ch = x_min + int(ww / 2), y_min + int(hh / 2)
            boyut = ww * hh            
            try:
                text = f'{classes[class_idx]} \n%{score*100:.0f}' 
            except: text = "wrong detection"
            
            #cv2.rectangle(frame, (x_min,y_min), (x_max, y_max), (0,255,0),1)
            cv2.circle(frame, (cw, ch), 8, (0,255,0), 2)
            alert = check_people_in_restricted_area(cw, ch, data, width, height)
            if alert:
                cv2.putText(frame, '!!! DIKKAT TEHLIKELI BOLGEDE INSAN VAR !!!', 
                            (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            # if (height * )
            for line in text.split('\n'):
                y_min += 30  
                cv2.putText(frame, line, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1) # font, font_scale, color, thickness)
        # general info:   
        cv2.putText(frame, 'Model: ' + model_path + ' Video:  ' + VIDEO_PATH, 
                    (5,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        # Distance info
        cv2.putText(frame, distance_text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if wrt:cv2.imwrite(out_write_path, frame)
        
        is_success, img_buffer = cv2.imencode(".png", frame)
        if is_success:
            io_buf = BytesIO(img_buffer)
            return Response(io_buf.getvalue(), media_type="image/png")
 
    else:
        cap.release()
        cap = None
        '''
        ALARM.write(False)
        ALARM.close()
        # Close the serial connection
        ser.close()
        '''
        return None
    

@app.post('/isg-tflite-model5x')
async def isg_tflite_model_test5x(data:LineValuesAndCheckboxes, request:Request=None):
    DETECTION_THRESHOLD = data.lineValues[4] / 100
    
    global cap, interpreter, VIDEO_PATH, model_path, frame_count, \
            trackers, scores, write_control, alert, det_line_pre
    if not data.start:
        try:
            VIDEO_PATH = "data/videos/" + data.selectedVideo
            cap.release()
            cap = None
        except Exception as e :print('Error', e)
        return None

    now = datetime.now()
    fileext = now.strftime("%H%M%S") + now.strftime("%f")[:3]
    out_write_path = f'{BASE_PATH}/data/logimg/{fileext}.jpg'
    in_write_path = re.sub('effi2ppe\d+\.tflite', f'in/{fileext}.jpg', model_path)
    
    wrt = False
    if cap is None:
        with open('data/initial.txt', 'w') as f: f.write('\n'.join(map(str, data.lineValues)))
 
        cap = cv2.VideoCapture(VIDEO_PATH)
        #roboflow 
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
    success, im = cap.read()
    
    if success:
        frame = cv2.resize(im, (960, 640))
        height, width, _ = frame.shape
        ## Measure distance
        #distance = read_distance(ser)
        if distance is not None:
            distance_text = f"Distance: {distance} cm"
        else:
            distance_text = "Distance: N/A"            
        if frame_count % (25 if alert else 5) == 0 :
            trackers.clear()
            input_size = get_input_size(interpreter)
            input_data = preprocess_input(frame.copy(), input_size=input_size)
            tt= time.time()        
            run_prediction(interpreter, input_data)
            detected_objects = extract_detected_objects(interpreter, min_score_thresh=DETECTION_THRESHOLD)
            print(f'Detection time {time.time()-tt:.3f}' )
            scores =[]
            bboxes = []
            for obj in detected_objects:
                # Convert coordinates from relative to absolute
                y_min, x_min, y_max, x_max = obj['box']
                x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
                ww, hh =( x_max-x_min, y_max - y_min ) 
                bboxes.append((x_min, y_min, ww, hh))
                
            scores = [obj['score'] for obj in detected_objects]
            trackers = [OPENCV_OBJECT_TRACKERS[trackerName]()  for _ in bboxes]
            for i, tracker in enumerate(trackers):
                ok = tracker.init(frame, bboxes[i])
                 
        
        for ii, tracker in  enumerate(trackers):
            # Convert coordinates from relative to absolute
            ok, bbox = tracker.update(frame)

            x_min, y_min, ww, hh = bbox
            
            wrt = True
            cw, ch = x_min + int(ww / 2), y_min + int(hh / 2)
            sqm2 = int(ww * hh)
            text = f'{classes[0]} \n%{scores[ii]*100:.0f}' 
            #cv2.rectangle(frame, (x_min,y_min), (x_max, y_max), (0,255,0),1)
            cv2.circle(frame, (cw, ch), 8, (0,255,0), 2)
            alert = check_people_in_restricted_area(cw, ch, data, width, height)
            if alert:
                det_line = f'{time.strftime("%d/%m/%Y %H:%M")} T. alanda {len(trackers)} kişi tespit edildi.#'
                if write_control and det_line_pre != det_line:
                    cv2.imwrite(out_write_path, frame)
                    file = time.strftime("%d%m%Y")
                    with open(f'data/logs/log{file[:]}.csv', 'a') as ff:
                        wr = writer(ff)
                        wr.writerow([det_line+fileext])
                        det_lines.append(det_line+fileext)
                        det_line_pre = det_line
            
                write_control = False
                cv2.putText(frame, '!!! DIKKAT TEHLIKELI BOLGEDE INSAN VAR !!!', 
                            (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            else: write_control = True
            # if (height * )
            for line in text.split('\n'):
                y_min += 30  
                cv2.putText(frame, line, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1) # font, font_scale, color, thickness)
        frame_count += 1 
        # general info:   
        cv2.putText(frame, 'Model: ' + model_path + ' Video:  ' + VIDEO_PATH, 
                    (5,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        # Distance info
        cv2.putText(frame, distance_text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        is_success, img_buffer = cv2.imencode(".png", frame)
        if is_success:
            img_base64 = base64.b64encode(img_buffer).decode("utf-8")
            response_data = {
                "image": img_base64,
                "det_lines": det_lines,
                "base_path": BASE_PATH
            }

            response_json = json.dumps(response_data)
            return Response(response_json, media_type="application/json")
            # io_buf = BytesIO(img_buffer)
            # return Response(io_buf.getvalue(), media_type="image/png")
    
    else:
        cap.release()
        cap = None
        '''
        ALARM.write(False)
        ALARM.close()
        # Close the serial connection
        ser.close()
        '''
        return None

def run_prediction(interpreter, input_data):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    detected_objects = extract_detected_objects(interpreter, min_score_thresh=DETECTION_THRESHOLD)
    # At the end of the function, put the detected_objects into the result_queue
    return detected_objects


            
# https://www.direnc.net/arduino-ultrasonic-mesafe-olcum-sensoru-urm37?language=tr&h=ef148c5f
def read_distance(ser):
    # Send the command to request distance data
    ser.write(b'\x22')  # 0x22 is the command for requesting distance in TTL mode

    # Read the response from the sensor
    response = ser.read(4)  # The response consists of 4 bytes

    if len(response) == 4 and response[0] == 0x22:
        # Extract the distance from the response
        distance = (response[1] << 8) | response[2]
        return distance
    else:
        # Invalid or incomplete response received
        return None

@app.get("/image/{image_name}")
async def serve_image(image_name: str):
    image_path = f"{BASE_PATH}/data/logimg/{image_name}.jpg"
    return FileResponse(image_path, media_type="image/jpeg")