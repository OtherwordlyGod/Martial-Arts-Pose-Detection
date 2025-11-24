# Martial-Arts-Pose-Detection
Using Mediapipe to run real time and video pose detection on martial art forms

In this project, I am using the google ai model Mediapipe, an open-source AI model specializing in tasks like pose estimation, to track and determine the pose of a subject. Specifically, the idea is to have karate practitioners perform their kata and upload the video where the pose would then be calculated and display through a series of lines of points on the body using the computer vision library OpenCv. The AI would also be able to display the correct kata name pulling from trained data. This program would also be able to work with things like free sparring though to a lesser degree of accuracy. 

This pose estimation works through two-step detector system that combines a computationally expensive object detector with a lightweight object tracker. The object detector is first ran and it creates a bounding box around the subject. Then the tracker comes in and predicts the position of points/landmarks inside the bounding box. As the video plays, the tracker continues to predict the landmarks, only reactivating the detector when it fails to track the person with high confidence. The model works the best when the subject is standing 2-4 meters away from the camera and it only works with single person pose-detection. The point or landmark positions that Mediapipe uses are shown below: 


<img width="1446" height="1712" alt="image" src="https://github.com/user-attachments/assets/9bbc2571-66cb-4177-8ff2-bb1e148a410d" />


0 - nose

1 - left eye (inner)

2 - left eye

3 - left eye (outer)

4 - right eye (inner)

5 - right eye

6 - right eye (outer)

7 - left ear

8 - right ear

9 - mouth (left)

10 - mouth (right)

11 - left shoulder

12 - right shoulder

13 - left elbow

14 - right elbow

15 - left wrist

16 - right wrist

17 - left pinky

18 - right pinky

19 - left index

20 - right index

21 - left thumb

22 - right thumb

23 - left hip

24 - right hip

25 - left knee

26 - right knee

27 - left ankle

28 - right ankle

29 - left heel

30 - right heel

31 - left foot index

32 - right foot index


Mediapipe uses exactly 33 unique points positioned across the body and connects these points with lines to form a skeleton of the pose.

This project was ultimately made for my 2026 Governor's Honors Program application in computer science which required me to build and showcase a project with around 2 months time. There are no requirements for .
I originally became interested in this idea because of me and my friend's shared love for martial arts. He happened to be a 1st degree black belt in American Tang Soo Do and had recently taken his black belt test which required the performance of various forms(kata). It was often an annoyance for me to practice martial arts at home, especially when imitating specific moves and techniques, without guidance. I decided to do some research on the idea of a form of motion capture that would let me extract the pose of someone performing a technique and grade its accuracy. During my research I happened upon A.I. pose estimation which looked very promising. 

Due to my limited resources and experience, A.I. pose estimation seemed like the solution to my problem as they were not particularly code-intensive to get started and there were a great number of resources on the internet that could help me. After some research, I decided upon Google's Mediapipe model because of its optimization for real-time performance and ease of use. 

Going into this, I had no experience with python or OpenCv(a computer vision library that is absolutely crucial to this project) which made it very difficult to advance past a certain point. To resolve this, I decided to utilize an AI assistant, specifically GPT-5.1, in order to help me work around python's syntax and Mediapipe in general.

#TODO
1. Add list of features
2. Add tech stack
3. Add visuals of project(example output)
4. Instructions on how to build/clone/run
5. Add diagram/flow description
6. Challanges and what I learned
7. Future updates

#ISSUES
1. Remove OneEuro filter

This project was started on 11-8-25 
