import math
import cv2 as cv
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from mediapipe.framework.formats import landmark_pb2


#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\kata-env\Scripts\activate


mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Pose (body) - higher complexity for better body accuracy
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)

# Hands - dedicated hand model (21 landmarks per hand)
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6)


POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# Drawing specs
POSE_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
POSE_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
HAND_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
HAND_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

def detect_and_draw(frame):


    # convert the coloring from BGR to RGB
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    image.flags.writeable = False


    # process the image and get the pose
    pose_results = pose.process(image)
    holistic_results = holistic.process(image)

    image.flags.writeable = True

    # make a copy of the frame to draw on
    out = frame.copy()

    # if person is detected and is analyzed, then call the mediapipe function draw_landmarks.
    # draw_landmarks takes 3 parameters being the image, the list of points, and the connections.
    # the pose and hands are draw seperately with the hands using the mp holistic model and the pose using mp pose model
    # Draw Pose (body only)
    if pose_results.pose_landmarks:
        
        mp_drawing.draw_landmarks(
            image=out,
            landmark_list=pose_results.pose_landmarks,
            connections=POSE_CONNECTIONS,
            landmark_drawing_spec=POSE_LANDMARK_SPEC,
            connection_drawing_spec=POSE_CONNECTION_SPEC
        )
        # Filter separately for scoring/logic
        filtered_landmarks = [lm for lm in pose_results.pose_landmarks.landmark if lm.visibility > 0.6]

    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=out,
            landmark_list=holistic_results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_SPEC,
            connection_drawing_spec=HAND_CONNECTION_SPEC
        )

    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=out,
            landmark_list=holistic_results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=HAND_LANDMARK_SPEC,
            connection_drawing_spec=HAND_CONNECTION_SPEC
        )


   # Optionally return raw landmark objects for downstream use (angles, scoring, etc.)
    return out, pose_results, holistic_results




video = cv.VideoCapture(r"C:\Users\other\codeProjects\Python\OpenCv\Garb test footage\appreciation form.mp4")
cv.namedWindow('Pose Detection', cv.WINDOW_NORMAL)


video.set(3, 1280)
video.set(4, 960)


time1 = 0


# This loops through every individual frame of the video and calls the detectPose function on each on
while video.isOpened():


    ok, frame = video.read()


    if not ok:
        break


    frame = cv.flip(frame, 1)
    out_frame, pose_res, hands_res = detect_and_draw(frame)


    cv.imshow('Pose + Hands', out_frame)
    if cv.waitKey(1) & 0xFF == 27:
        break


video.release()
cv.destroyAllWindows()


 

