import cv2
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import webcolors

def get_color_name(rgb_value):
    min_distance = float('inf')
    closest_color = None

    for hex_color, color_name in webcolors.CSS3_NAMES_TO_HEX.items():
        try:
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_color)
        except ValueError:
            continue

        rd = (r_c - rgb_value[0]) ** 2
        gd = (g_c - rgb_value[1]) ** 2
        bd = (b_c - rgb_value[2]) ** 2
        distance = rd + gd + bd

        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def get_limits(color):
    if color == 'red':
        lower_limit = np.array([0, 50, 50])
        upper_limit = np.array([10, 255, 255])
    elif color == 'green':
        lower_limit = np.array([36, 50, 50])
        upper_limit = np.array([70, 255, 255])
    elif color == 'white':
        lower_limit = np.array([0, 0, 200])
        upper_limit = np.array([180, 30, 255])
    elif color == 'black':
        lower_limit = np.array([0, 0, 0])
        upper_limit = np.array([180, 255, 30])
    else:
        raise ValueError(f"Invalid color: {color}")

    return lower_limit, upper_limit


def display_color_name(frame, color_name):
    cv2.putText(frame, color_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colors = ['red', 'green', 'white', 'black']
    for color in colors:
        lowerLimit, upperLimit = get_limits(color=color)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # Extract the region of interest within the bounding box
            roi = frame[y1:y2, x1:x2]

            # Reshape the ROI to a 2D array of pixels
            pixels = roi.reshape(-1, 3)

            # Perform K-means clustering to identify the dominant color
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(int)

            # Convert BGR to RGB color format
            rgb_color = [dominant_color[2], dominant_color[1], dominant_color[0]]

            # Get the color name
            color_name = get_color_name(rgb_color)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            frame = display_color_name(frame, color_name)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
