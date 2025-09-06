import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import os
import time
from google.protobuf.json_format import MessageToDict

# Initialize MediaPipe Hands Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)

# Start video capture
cap = cv2.VideoCapture(0)

# Variables for cooldown
last_gesture_time = 0
cooldown_time = 20  # Cooldown time in seconds

# Functions to detect specific gestures
def detect_two_fingers_up(hand_landmarks):
    # Detect if thumb and index fingers are pointing upwards, and the other fingers are curled.
    thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_IP]
    
    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP]

    # Check if thumb and index finger are pointing up
    if thumb_tip.y < thumb_ip.y and index_tip.y < index_dip.y:
        return True
    return False

def detect_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP]
    
    if (index_tip.y > hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y and
        middle_tip.y > hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y and
        ring_tip.y > hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_MCP].y and
        pinky_tip.y > hand_landmarks.landmark[mpHands.HandLandmark.PINKY_MCP].y):
        return True
    return False

def detect_pointing_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
    
    if index_tip.y < index_mcp.y:
        return True
    return False

# Start capturing video from webcam
while True:
    # Read frame from webcam
    success, img = cap.read()
    
    # Flip the image horizontally
    img = cv2.flip(img, 1)
    
    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe hands
    results = hands.process(imgRGB)
    
    # Get current time to check cooldown
    current_time = time.time()

    # Check if enough time has passed since last gesture (20 seconds cooldown)
    if current_time - last_gesture_time >= cooldown_time:
        # Check if hands are detected
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification (left or right)
                label = MessageToDict(results.multi_handedness[i])['classification'][0]['label']
                
                if label == 'Left':
                    # Handle left hand gestures
                    if detect_fist(hand_landmarks):
                        print("Left Hand: Closing the Gesture Program")
                        cv2.destroyAllWindows()  # Close the gesture detection window
                        cap.release()  # Release the webcam
                        break  # Exit the loop to stop the program

                elif label == 'Right':
                    # Handle right hand gestures
                    if detect_two_fingers_up(hand_landmarks):
                        print("Right Hand: Opening Spotify")
                        webbrowser.open("https://open.spotify.com")  # Open Spotify in the default browser
                        last_gesture_time = current_time  # Update the last gesture time
                        break  # Exit loop to process gesture once
                    
                    if detect_fist(hand_landmarks):
                        print("Right Hand: Muting")
                        pyautogui.press('volumemute')  # Mute the system
                        last_gesture_time = current_time  # Update the last gesture time
                        break  # Exit loop to process gesture once

                    if detect_pointing_up(hand_landmarks):
                        print("Right Hand: Raising volume")
                        pyautogui.press('volumeup')  # Increase volume
                        last_gesture_time = current_time  # Update the last gesture time
                        break  # Exit loop to process gesture once

        # Get the image height for positioning text at the bottom left
        img_height = img.shape[0]

        # Resize the font and adjust position to the bottom-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Smaller font size
        color = (0, 0, 0)  # Black color
        thickness = 1  # Thinner text

        # Add the gesture instructions at the bottom left corner
        cv2.putText(img, "Gestures:", (10, img_height - 150), font, font_scale, color, thickness)
        cv2.putText(img, "Two Fingers Up (Right Hand): Open Spotify", (10, img_height - 110), font, font_scale, color, thickness)
        cv2.putText(img, "Fist (Right Hand): Mute", (10, img_height - 70), font, font_scale, color, thickness)
        cv2.putText(img, "Fist (Left Hand): Close the Program", (10, img_height - 30), font, font_scale, color, thickness)
        cv2.putText(img, "Pointing Up (Right Hand): Volume Up", (10, img_height - 10), font, font_scale, color, thickness)
    else:
        # If cooldown hasn't passed, display a message
        cv2.putText(img, "Please wait for 20 seconds...", (10, 50), font, font_scale, (0, 0, 0), thickness)

    # Display the image with the instructions
    cv2.imshow('Hand Gesture Control', img)
    
    # If 'q' is pressed, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
