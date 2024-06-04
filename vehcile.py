import cv2
import numpy as np
import time

# Constants
MIN_WIDTH_RECT = 80  # Minimum width of rectangle
MIN_HEIGHT_RECT = 80  # Minimum height of rectangle
LINE_POSITION = 550  # Position of the detection line
OFFSET = 6  # Offset for line detection
DELAY = 0.1  # Delay in seconds

# Variables
detected_centers = []
vehicle_count = 0

# Function to calculate the center of the rectangle
def get_center(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

# Load video
cap = cv2.VideoCapture('video.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Loop through frames in the video
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Introduce delay to slow down video playback
    time.sleep(DELAY)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 5)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(blurred_frame)

    # Apply morphological operations to remove noise
    dilated_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8))
    dilated_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a line on the frame
    cv2.line(frame, (25, LINE_POSITION), (1200, LINE_POSITION), (255, 127, 0), 3)

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= MIN_WIDTH_RECT and h >= MIN_HEIGHT_RECT:
            # Draw rectangle around detected vehicle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center of the rectangle
            center = get_center(x, y, w, h)
            detected_centers.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    # Check if detected center crosses the line
    for center in detected_centers:
        cx, cy = center
        if LINE_POSITION - OFFSET < cy < LINE_POSITION + OFFSET:
            vehicle_count += 1
            cv2.line(frame, (25, LINE_POSITION), (1200, LINE_POSITION), (0, 127, 255), 3)
            detected_centers.remove(center)
            print(f"Vehicle detected: {vehicle_count}")

    # Display vehicle count on the frame
    cv2.putText(frame, f"VEHICLE COUNT: {vehicle_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display the resulting frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Detection Mask", dilated_mask)

    # Exit if 'e' is pressed
    if cv2.waitKey(1) == ord('e'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
