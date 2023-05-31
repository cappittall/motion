import copy
import itertools
import cv2
import numpy as np
from PIL import Image
import supervision as sv

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

def is_using_cellphone(face_landmarks, hand_landmarks, mp_hands, cellphone_distance_threshold=0.2):
    # Define a threshold distance for cell phone usage detection


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



def text_update(text, smoking_detected, mp_cellphone_usage_detected, cellphone_usage_detected, emotion):
        text += " | Sigara: VAR " if smoking_detected else " | Sigara: YOK "
        text += f" | Tel: VAR.({mp_cellphone_usage_detected})" if cellphone_usage_detected else f" | Tel: YOK.({mp_cellphone_usage_detected})"
        text += f"| Ruhsal D: {emotion}"
        return text
    
def yolo_detect_n_annotate_imame(image, yolo_model, nas_model, nas=True):
    
    if nas:
        result = list(nas_model.predict(Image.fromarray(image), conf=0.35))[0]
        detections = sv.Detections(
                    xyxy=result.prediction.bboxes_xyxy,
                    confidence=result.prediction.confidence,
                    class_id=result.prediction.labels.astype(int)
                )
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{result.class_names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections]
        annotated_frame = box_annotator.annotate(
                        scene=image.copy(),
                        detections=detections,
                        labels=labels
                    )
        
        return annotated_frame
    
    else:
        detections = yolo_model.predict(Image.fromarray(image))
        for detection in detections:           
            # Get bounding box coordinates and convert them to integers
            (x1, y1, x2, y2), class_index, conf = map(int, detection.boxes.xyxy[0]), int(detection.boxes.cls), int(detection.boxes.conf * 100)
            # Draw the bounding box and label only for cell phones (class index 67)
            if class_index == 67:
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"cep tel {conf}%", (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),1)
                # Get the class label from the class index
                cellphone_usage_detected = True
                # Draw the class label and confidence score
        return image


    