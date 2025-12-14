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
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)

# Hands - dedicated hand model (21 landmarks per hand)
holistic = mp_holistic.Holistic(static_image_mode=True,
                                model_complexity=1,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6)


POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# Drawing specs
POSE_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)
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

    # Draw Pose with pose model.
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

    # Draw hands using holistic model
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

# Helper function to clamp the cos value    
def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

# Calculates using vector dot product formula
def calculate_angle(a, b, c): 
    if (min(a.visibility, b.visibility, c.visibility) > 0.6):
        dot_product = (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) + (a.z - b.z) * (c.z - b.z)

        mag_BA = math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        mag_BC = math.sqrt((c.x - b.x)**2 + (c.y - b.y)**2 + (c.z - b.z)**2)

        # Prevents division by 0
        if mag_BA == 0 or mag_BC == 0:
            return None

        # Calculates the angle in radians
        angle_rad = math.acos(clamp((dot_product / (mag_BA * mag_BC)), -1, 1))

        # Converts to degree
        return (angle_rad * 180) / math.pi
    else:
        return None

def calculate_pose_angles(results):

    # Prevents crashing when no person is detected
    if (results.pose_landmarks):

        right_elbow = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                                    )

        right_shoulder = calculate_angle(
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
                                        )

        right_hip = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                                    )

        right_knee = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                                    )
    
        right_ankle = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                                    )

        left_elbow = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                                    )

        left_shoulder = calculate_angle(
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
                                    )

        left_hip = calculate_angle(
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
                                )

        left_knee = calculate_angle(
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                                )
    
        left_ankle = calculate_angle(
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                                )

        pose_angles = [right_elbow, right_shoulder, right_hip, right_knee, right_ankle, left_elbow, left_shoulder, left_hip, left_knee, left_ankle]

    return pose_angles


image = cv.imread(r"C:\Users\other\codeProjects\Python\Stances\front stance\front stance 1.png")

if image is None:
    print("Failed to load image")
    exit()

# Convert to RGB
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Pose detection
out_image, pose_res, hands_res = detect_and_draw(image)

angles = calculate_pose_angles(pose_res)

print("\nPOSE ANGLES:")
if angles:
    for joint, angle in zip(
        ["R_elbow", "R_shoulder", "R_hip", "R_knee", "R_ankle",
         "L_elbow", "L_shoulder", "L_hip", "L_knee", "L_ankle"],
        angles
    ):
        if (angle != None):
            print(f"{joint}: {angle:.2f}°")
        else: 
            print(f"{joint}: None")
else:
    print("No pose detected.")


# Show image
cv.imshow("Stance Pose", out_image)
cv.waitKey(0)
cv.destroyAllWindows()



# video = cv.VideoCapture(r"C:\Users\other\codeProjects\Python\Garb test footage\intermediate form 4.mp4")
# cv.namedWindow('Pose Detection', cv.WINDOW_NORMAL)


# video.set(3, 1280)
# video.set(4, 960)


# pre_timeframe = 0
# new_timeframe = 0


# # This loops through every individual frame of the video and calls the detectPose function on each on
# while video.isOpened():


#     ok, frame = video.read()


#     if not ok:
#         break

    
#     frame = cv.resize(frame, (1280, 720))

#     new_timeframe = time.time()
#     fps = 1/(new_timeframe - pre_timeframe)
#     pre_timeframe = new_timeframe
#     fps = int(fps)
#     cv.putText(frame, str(fps), (8, 80), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 0, 255), 4)

#     out_frame, pose_res, hands_res = detect_and_draw(frame)

#     cv.imshow('Pose + Hands', out_frame)
#     if cv.waitKey(1) & 0xFF == 27:
#         break

# # Prints frame time
# start = time.time()
# pose.process(frame)
# holistic.process(frame)
# print("FRAME TIME", time.time() - start)

# video.release()
# cv.destroyAllWindows()


 

