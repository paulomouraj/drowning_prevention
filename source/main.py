import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2

# *****BUTTONS GPIO CONFIG*****
# Reading GPIO 23
But1 = 16
# Use board GPIO pin
GPIO.setmode(GPIO.BOARD)
# Set off the warnings
GPIO.setwarnings(False)
# But1 and But2 as input and pull-up
GPIO.setup(But1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# Resets pool calibration flag
flag_calibrate = 0

# Callback function for button press


def handle(pin):
    if pin == But1:
        global flag_calibrate
        flag_calibrate = 1
        print("Calibrating...")


# External interrupt for button press
GPIO.add_event_detect(But1, GPIO.FALLING, handle, bouncetime=300)

# Function to detect the pool area


def detectPool(image):
    # Resets the flag that indicates whether contours were detected or not
    flag_contours = 0
    # Creates a blank image of the same size as the original
    edges = np.zeros([480, 640, 3], np.uint8)
    # Defines the lower and upper color limits for mask creation, here set as blue
    blue_lower = np.array([100, 90, 40])
    blue_upper = np.array([140, 255, 255])
    # Applies a Gaussian filter to smooth the image
    image = cv2.GaussianBlur(image, (11, 11), 0)
    # Converts the colorspace to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Creates a binary mask, keeping the pixels of interest
    mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
    # Applies morphological filters to the mask to remove noise - closing and opening
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Looks for all contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Gets the contours to compare sizes later
    all_contours = [(cv2.contourArea(contour), contour)
                    for contour in contours]
    # Checks if there is a contour and draws the largest one
    if (len(all_contours) > 0):
        # Sets the contours flag, indicating that a pool was found
        flag_contours = 1
        # Displays that the pool was successfully recognized and calibrated
        print("Calibrated successfully.")
        largest_contour = max(all_contours, key=lambda x: x[0])[1]
        # Draws the pool on the blank image
        cv2.drawContours(edges, [largest_contour], -1, (0, 0, 255), 10)
    else:
        # Resets the contour variable
        largest_contour = 0
        print("Failed to identify the pool.")

    # Returns the largest contour to be drawn on the image stream
    return flag_contours, edges, largest_contour

# Function that checks if a person is close to the pool


def verify(edges, xA, yA, xB, yB):
    flag = 0
    if yB >= edges.shape[0]:
        yB = edges.shape[0] - 1
    if xB >= edges.shape[1]:
        xB = edges.shape[1] - 1

    j = 0
    while (j <= (xB - xA)):
        if (edges.item(yA, (xA + j), 2) == 255):
            flag = 1
            break
        if (edges.item(yB, (xA + j), 2) == 255):
            flag = 1
            break
        j += 1

    j = 0
    while (j <= (yB - yA)) and (flag == 0):
        if (edges.item((yA + j), xA, 2) == 255):
            flag = 1
            break
        if (edges.item((yA + j), xB, 2) == 255):
            flag = 1
            break
        j += 1

    return flag


# Import the MobileNet SSD model with the pre-trained weights
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# *****PI CAMERA CONFIG*****
camera = PiCamera()
camera.resolution = (640, 480)
camera.rotation = 180
rawCapture = PiRGBArray(camera)

# Create the MOG2 background subtractor object
filter = cv2.createBackgroundSubtractorMOG2()

# Resets pool detection flag
flag_pool = 0

# Setting MOG2 trheshold values
area_total = 160*120
area_min = 0.1*area_total
area_max = 0.3*area_total

# Setting the min softmax threshold for person detection to avoid false positives
min_prob = 0.01

# Loop to capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Converts the frame to an image
    img = frame.array

    # *****POOL IDENTIFICATION*****
    # If the calibration button is pressed
    if flag_calibrate == 1:
        # Sends the frame to the function that will recognize the pool
        flag_pool, pool, pool_contour = detectPool(img)
        # Resets the calibration flag
        flag_calibrate = 0

    # *******DRAWING THE POOL ON THE FINAL IMAGE*******
    # If the pool is identified, draw it on the final display image before any detection boxes
    if flag_pool == 1:
        img_final = img
        cv2.drawContours(img_final, [pool_contour], -1, (0, 0, 255), 10)

    # *******MOG2*********
    # Resizes the original image
    img_mog = cv2.resize(img, (160, 120))
    # Create the foreground mask using MOG2
    foregrnd_mask = filter.apply(img_mog)
    # Threshold the mask to remove shadows
    _, foregrnd_mask = cv2.threshold(
        foregrnd_mask, 254, 255, cv2.THRESH_BINARY)
    # Apply morphological filters to the mask to remove noise - closing and opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    foregrnd_mask = cv2.morphologyEx(foregrnd_mask, cv2.MORPH_OPEN, kernel)

    # *****PERSON DETECTION****--> Pre-trained MobileNetSSD CNN
    # If the pool is calibrated --> To avoid overloading the Raspberry Pi, the neural network only works when the pool is calibrated
    if flag_pool == 1:
        # Find all contours in the MOG2 mask
        # Loop through the contours detected in the MOG2 mask
        conts, _ = cv2.findContours(
            foregrnd_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Reset the detection flag
        flag_detect = 0
        for cont in conts:
            # Ignore image noise and very large objects to avoid freezing the code
            if (cv2.contourArea(cont) < area_min) or (cv2.contourArea(cont) > area_max):
                continue
            # Generate rectangles that bound the contour and return the rectangle's coordinates *Starting point (x, y), width (w), and height (h)
            (x, y, w, h) = cv2.boundingRect(cont) * np.array([4, 4, 4, 4])
            # Crop the image based on the contour area
            region_of_interest = img[y:y+h, x:x+w]
            # Create a blob to input into the neural network
            blob = cv2.dnn.blobFromImage(
                region_of_interest, 0.007843, (w, h), 127.5)
            # Input the blob into the neural network
            net.setInput(blob)
            # Forward pass to get the detections
            detects = net.forward()
            # Reset the alarm flag
            flag_alarm = 0
            # Loop through detections if there are any
            for i in np.arange(0, detects.shape[2]):
                prob = detects[0, 0, i, 2]
                detection_class = int(detects[0, 0, i, 1])
                # Check if it's a person --> Class 15 and if the probability is higher than the min threshold
                if (prob > min_prob and detection_class == 15):
                    flag_detect = 1
                    # getting full coordinates of the bounding box
                    xR = x + w
                    yR = y + h
                    # Check if the detection box intersects with the pool edge
                    flag_alarm = verify(pool, x, y, xR, yR)
                    # Draw the area where people were detected for visualization
                    cv2.rectangle(img_final, (x, y), (xR, yR), [0, 255, 0], 2)
                    # If the detection box intersects the pool edge, exit the detection loop
                    if flag_alarm == 1:
                        break
            # If the detection box intersects the pool edge, exit the contours loop
            if flag_alarm == 1:
                break

        # DISPLAY THE CURRENT STATUS ON SCREEN FOR EACH CASE
        if flag_detect == 1 and flag_alarm == 1:
            cv2.putText(img_final, "INSIDE POOL AREA", (200, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA)
            cv2.imshow("Last detection", img_final)
            cv2.imshow("Detection", img_final)
            # If a person is detected inside the pool area, sound the alarm
            print("ALARM! A person is inside the pool area.")
        if flag_detect == 1 and flag_alarm == 0:
            cv2.putText(img_final, "OUTSIDE POOL AREA", (200, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA)
            cv2.imshow("Detection", img_final)
        if flag_detect == 0:
            cv2.putText(img_final, "NO DETECTIONS", (200, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA)
            cv2.imshow("Detection", img_final)
    else:
        # If the pool edge is not calibrated, display the raw frame
        cv2.putText(img, "WAITING FOR POOL CALIBRATION", (100, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow("Detection", img)

    # Clear the stream for the next frame
    rawCapture.truncate(0)

    # If Q is pressed exit the running loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
camera.close()
