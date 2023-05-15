import copy
import itertools
import cv2
import numpy as np


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

def is_using_cellphone(face_landmarks, hand_landmarks, mp_hands):
    # Define a threshold distance for cell phone usage detection
    cellphone_distance_threshold = 0.5

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