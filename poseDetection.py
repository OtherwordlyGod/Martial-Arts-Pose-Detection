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

#initializing mp pose class
mp_pose = mp.solutions.pose

#setting up the pose function
pose_video = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.7, min_tracking_confidence = 0.7, model_complexity = 1)

#initializing mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils


# OneEuro implementation (single coordinate)
class OneEuro:
    def __init__(self, init_val=0.0, freq=30.0, min_cutoff=0.005, beta=0.5, d_cutoff=1.0):
        self.x_prev = init_val
        self.dx_prev = 0.0
        self.t_prev = None
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp):
        if self.t_prev is None:
            self.t_prev = timestamp
            self.x_prev = x
            return x
        # compute derivative
        dt = timestamp - self.t_prev
        self.freq = 1.0 / dt if dt>0 else self.freq
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # adapt cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        # update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat

# each of the 33 landmarks need their own set of filters therefore here we make 99 instances of OneEuro
filters = [

    {
        "x" : OneEuro(), 
        "y" : OneEuro(), 
        "z" : OneEuro(),
    }

    for _ in range(33)
]


def detectPose(image, pose, timestamp, display = True): 

    # copy the image to make sure we are not drawing on the orignal
    output_image = image.copy()

    # convert the coloring from BGR to RGB
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # process the image and get the pose
    results = pose.process(imageRGB)

    # calculate dimensions of image
    height, width, _ = image.shape

    # a list to store the points in later
    landmarks = []

    # if person is detected and is analyzed, then call the mediapipe function draw_landmarks.
    # draw_landmarks takes 3 parameters being the image, the list of points, and the connections.
    if results.pose_world_landmarks: 

        # store all points in landmarks list and by calling enumerate, i is automatically incremented 0 -> 32.
        for i, landmark in enumerate(results.pose_landmarks.landmark):         
        # We smooth out the x, y, and z coords of each landmark by calling the filters list and using i to go from the first landmark to the last, filtering each one.
            smoothed_x = filters[i]["x"].filter(landmark.x, timestamp)
            smoothed_y = filters[i]["y"].filter(landmark.y, timestamp)
            smoothed_z = filters[i]["z"].filter(landmark.z, timestamp)

            landmarks.append((smoothed_x, smoothed_y, smoothed_z))

        smoothed_landmark_list = landmark_pb2.NormalizedLandmarkList()

        for (x, y, z) in landmarks: 

            # landmark_pb2 comes form the Protocol Buffers which mediapipe uses to define its date structures. 
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = x
            lm.y = y 
            lm.z = z 
            # .landmark is the container for landmarks in a NormalizedLandmarkList
            smoothed_landmark_list.landmark.append(lm)

        mp_drawing.draw_landmarks(
            image = output_image, 
            landmark_list = smoothed_landmark_list, 
            connections = mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 5, circle_radius = 2), 
            connection_drawing_spec = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 3, circle_radius = 1)
        )
    
    # to display or not to display, that is the question
    if display: 

        # creates a matpplotlib to show results
        plt.figure(figsize = [22, 22])

        # creates 1 row and 2 collumns to compare original and output side by side.
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output image");plt.axis('off');

        # shows the 3d landmark plot
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    
    else: 
        return output_image, landmarks

video = cv.VideoCapture(r"C:\Users\other\codeProjects\Python\OpenCv\Garb test footage\No Gi test 2.MOV")
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

    frame_height, frame_width, _ = frame.shape

    frame = frame = cv.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    current_time = time.time()
    frame, _ = detectPose(frame, pose_video, current_time, display = False)

    cv.imshow('Pose', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

video.release()
cv.destroyAllWindows()

 