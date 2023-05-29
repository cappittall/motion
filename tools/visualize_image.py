
import base64
from collections import Counter
import numpy as np
import cv2 
import matplotlib.pyplot as plt

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text =  f' Cep Tel ({probability})'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

def _get_emotion_chart(dominant_emotions, emotion_data, emotion_counts, current_time, size):
    emotion_counter = Counter(emotion_data)
    dominant_emotion = emotion_counter.most_common(1)[0][0]  # most common emotion
    dominant_emotions.append((current_time.strftime("%H:%M"), dominant_emotion))
    emotion_counts[dominant_emotion] += 1  # increment count of dominant emotion
    shift_start_time = current_time 
    # create/update the chart
    unique_emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    fig, ax = plt.subplots()
    ax.bar(unique_emotions, counts, align='center', alpha=0.5)
    ax.set_ylabel('Counts')
    ax.set_title('Dominant Emotions Throughout the Work Day')
    plt.savefig('emotion_chart.png')
    plt.close(fig)

    # encode image to base64
    with open('emotion_chart.png', 'rb') as img_file:
        emotion_chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Add chart to the main image
    emotion_chart = cv2.imdecode(np.frombuffer(base64.b64decode(emotion_chart_base64), dtype=np.uint8), 1)
    emotion_chart = cv2.resize(emotion_chart, size) 
    return emotion_chart, dominant_emotions, emotion_data, emotion_counts, current_time, shift_start_time

def get_emotion_chart(dominant_emotions, emotion_data, emotion_counts, current_time, size):
    emotion_counter = Counter(emotion_data)
    dominant_emotion = emotion_counter.most_common(1)[0][0]  # most common emotion
    dominant_emotions.append((current_time.strftime("%H:%M"), dominant_emotion))
    emotion_counts[dominant_emotion] += 1  # increment count of dominant emotion
    shift_start_time = current_time 
    # create/update the chart
    unique_emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    # Create a color list, one for each unique emotion
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # add more colors if there are more than 7 unique emotions
    color_list = color_list[:len(unique_emotions)]  # truncate list to the number of unique emotions

    fig, ax = plt.subplots()

    # Add a small space between the bars
    ax.bar(unique_emotions, counts, align='center', alpha=0.5, color=color_list, width=0.8)

    ax.set_ylabel('Sayi', fontsize=12)  # Increase font size
    ax.set_title('Baskin duygusal veri grafigi', fontsize=14)  # Increase font size

    # Increase font size and rotate labels
    plt.xticks(fontsize=20) #, rotation=30)  # Use rotation only if labels are too long
    plt.yticks(fontsize=20)

    # Set the background color of the chart to be transparent
    fig.patch.set_alpha(0.0)  # this sets transparency of the figure
    ax.patch.set_alpha(0.0)  # this sets transparency of the plot

    # Save the chart as a .png file, which supports transparency
    plt.savefig('emotion_chart.png', transparent=True)
    plt.close(fig)

    # encode image to base64
    with open('emotion_chart.png', 'rb') as img_file:
        emotion_chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # Add chart to the main image
    # Ensure to read the image with alpha channel (transparency)
    emotion_chart = cv2.imdecode(np.frombuffer(base64.b64decode(emotion_chart_base64), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    emotion_chart = cv2.resize(emotion_chart, size) 

    return emotion_chart, dominant_emotions, emotion_data, emotion_counts, current_time, shift_start_time

def bind_emotion_chart_on_image(image, emotion_chart):                  
        # Convert base image to RGBA
        if image.shape[2] == 3:  # RGB image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Define where to place emotion_chart on image
        start_y = 0
        end_y = start_y + emotion_chart.shape[0]
        start_x = 0
        end_x = start_x + emotion_chart.shape[1]

        # Extract alpha channel from emotion_chart
        alpha_s = emotion_chart[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # Do alpha blending
        for c in range(0, 3):
            image[start_y:end_y, start_x:end_x, c] = (alpha_s * emotion_chart[:, :, c] +
                                        alpha_l * image[start_y:end_y, start_x:end_x, c])
        return image
    