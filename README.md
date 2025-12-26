# Martial-Arts-Pose-Detection
Using Mediapipe to run real time and video pose estimation and classification on American Tang Soo Do forms.

This project was made for my 2026 Governor's Honors Program application in computer science which required me to build and showcase a project with around 2 months time. 
I originally became interested in this idea because of me and my friend's shared love for martial arts. He happened to be a 1st degree black belt in American Tang Soo Do and had recently taken his black belt test which required the performance of various forms(kata). 

Being students and all, it was impossible for us to be at the gym 24/7 so a large portion of our time training martial arts took place at home without guidance. The lack of a coach guiding out movements made imitating specific moves and techniques very difficult to get right. On top of that, the lack of guidance could result in us building bad habits such as and not limited to dropping your guard hand, pushing punches, and not rotating through. 

As the resident computer scientist of the group, I decided to do some research on the idea of a form of motion capture that would let me extract the pose of someone performing a technique as a potential solution to our problem. During my research I happened upon something called A.I. pose estimation which looked very promising. Considering my limited resources and experience, A.I. pose estimation seemed like the solution to my problem as it was not particularly code-intensive to get started and there were a great number of resources on the internet that could help me. After some research, I decided upon Google's Mediapipe model because of its optimization for real-time performance and ease of use. 

Going into this, I had no experience with python or OpenCv which made it very difficult to advance past a certain point. To resolve this, I decided to utilize an AI assistant, specifically GPT-5.1, in order to help me with python's syntax and Mediapipe in general. I also recieved assistance in the form of various articles, papers, and blog posts all covering the idea of using pose estimation within martial arts. Of course throughout my journey I was careful not to rely too much on external help as it would negatively impact my learning but for certain situations it was unavoidable as much of this project lied outside my skillset. 

In this project, I am using BlazePose through the google ai framework Mediapipe, an open-source AI model specializing in tasks like pose estimation, to track and classify the pose of a subject. Specifically, the idea is to have karate practitioners perform their kata and upload the video where the pose would then be calculated and display through a series of lines of points on the body drawn by the computer vision library OpenCv. The AI would also be able to display the correct kata name pulling from trained data. This program would also be able to work with things like free sparring though to a lesser degree of accuracy. As of right now, the prototype is able to track the subject and constantly draw landmarks on its body while also calculating angles and classifying body pose, checking if it matches any supported stances.

This pose estimation works through two-step detector system that combines a computationally expensive object detector with a lightweight object tracker. The object detector is first ran and it creates a bounding box around the subject. Then the tracker comes in and predicts the position of points/landmarks inside the bounding box. As the video plays, the tracker continues to predict the landmarks, only reactivating the detector when it fails to track the person with high confidence. The model works the best when the subject is standing 2-4 meters away from the camera and it only works with single person pose-detection. The point or landmark positions that Mediapipe uses are shown below: 


<img width="850" height="958" alt="image" src="https://github.com/user-attachments/assets/1a00d51a-f359-447b-b78f-bc0ede5cb09c" />


I originally wanted to combine Mediapipe's pose estimation model with its holistic model for a total of 74 unique points across the body: 21 landmarks on each hand and the remaining 33 points spread across the body at key joints. However it was found that layering both models causes a significant decrease in performance with fps dropping from ~30 to less than 12 frames a second. This lag made it incredibly difficult to run test footage through which and in order to avoid further performance issues, I only used the lighter pose model for my first minimal viable product.

The project contains a fully functional pose detection and classification pipeline, alongside a real-time video processing loop capable of applying MediaPipe’s pose estimation A.I. to both prerecorded video and live camera feeds. Mediapipe's pose detection module supports three different model complexities: fast, medium, and heavy with each having its own trade off between computational cost and landmark accuracy. The fast model prioritizes speed over accuracy and is best suited for real-time applications on lower-end hardware while the heavy model focuses on landmark accuracy and is built for deep analysis on more powerful systems. In my project, I mainly used the medium model complexity as it strikes a good balance between the two and is more versatile.

The pose classification system currently supports 3 Tang Soo Do stances: June Bi(Starting stance), Chun Kul Cha Seh(Front stance), and Dwi Kubi(Back Stance). The original concept also included Neko Ashi Dachi(Cat stance) and Kee Ma Jaseh(Horse stance) but I unfortunately lacked the data to implement these stances accurately. Each stance is defined by a set of acceptable joint angle ranges, not fixed values, to allow for natural variations during movement. These ranges were derived by averaging angle measurements collected from multiple reference images of the same stance and the number sets were stored in the pose angles.txt file.

The joint angles are measured and calculated using vector mathematics, specifically using the dot product and the law of cosines. For each joint, two vectors are constructed from three key pose landmarks. For example, to compute the right knee angle (angle B), vectors are formed between the right hip (A) → right knee (B) and the right ankle (C) → right knee (B). The angle at the joint is then defined as the angle between vectors BA and BC. Angle theta(B) is calculated through the dot product formula: θ = arccos.((vector BA * vector BC) / magnitude BA * magnitude BC). Each vector is constructed using the difference between landmark coordinates: (x2−x1, y2−y1, z2−z1) and to find the magnitudes, you can use sqrt(x^2+y^2+z^2). Before applying the dot product formula, the normalized dot product is clamped to the range [-1, 1] to prevent errors. Finally the resulting angle is converted into degrees for easier readability. By using vector mathematics in three-dimensional space, the pose is modeled as a collection of 3D points which ensures the joint angles are rotational invarient and scale indepedent, making them respond to camera orientation, subject distance, and body proportions. To improve reliability, the joint angles are filtered my Mediapipes' landmark visibility score which removes noise and prevents partially occluded body parts from misleading results.

Stance classification works by comparing the current frame’s angle array(elbows, shoulders, hips, knees, and ankles) to predefined stance ranges. Each valid joint comparison contributes to an accuracy score that is then divided by the maximum possible score to get a confidence percentage. If the confidence score is above a certain threshold, the appropriate stance is returned. To handle left–right symmetry (e.g., mirrored stances), the system evaluates both the original and reversed angle arrays, allowing the stance to be recognized regardless of which leg is forward.

Because the pose estimation over individual frames can be incredibly noisy or inconsistent, I implemented temporal smoothing through a five frame buffer for stance classification. Rather than displaying the stance detected in a single frame, the most frequently appearing stance classification within the buffer period is displayed. This majority-vote approach significantly reduces flickering and improves the overall smoothness, especially for stances that appear briefly or during transitions.

Example Output: 
![alt text](<example output/ready stance classified 2.png>)
![alt text](<example output/front stance classified 1.png>)
![alt text](<example output/back stance classified 2.png>)

The curerent version of this project is merly a prototype and lacks many of the features I orignally wanted to implement. I do plan on continuing this project regardless of what GHP thinks of it and the overall vision is to turn this into a mobile application that uses the phone camera and a live video feed to record a practitioner and grade their stance accuracy and giving live feedback. I also had the idea to implement this technology into combat sports like MMA, Taekwondo, Wrestling, and Boxing where the A.I. could be trained to track strikes and judge fights. However before I am even close to that dream, there are a lot of improvements that can be made to the current prototype.
Some of my ideas are listed below:

TODO:
1. Add more supported stances
2. Implement holistic model to track hands without hurting performance 
3. Add pose recognition and classification for basic punches and blocks
4. Implement a filter to reduce landmark jitter
5. Add multi-person pose estimation using models like YOLO
6. Add time controls (e.g. pause, replay, etc)
7. Add a user interface 
8. Recognize and classify entire form sequences
9. Grade stance/pose accuracy

This is my biggest project to date and in the 48 days it took to build this prototype, I went from having virtually zero python and computer vision experiance to building a project using state of the art A.I. framework. This project has given me the opportunity to be exposed to industry leading libraries like OpenCv, powerful frameworks like Mediapipe, and advanced mathmatic concepts like vector mathmatics which are all useful concepts to know moving foreward into this field. On top of being incredibly ambitious, this project was also incredibly difficult for me to develop as I had limited knowledge of python syntax and Mediapipe attributes. I ended up relying on outside help such as A.I. or youtube tutorials to get the code for parts I understood conceptually but had not yet learned to translate into working code. All in all, I am very proud of what I was able to accomplish here and I look forward to what this project will become in the future. 

Tech Stack: 

Core Language

Python 3 
primary language for pose processing, numerical computation, and video handling

Computer Vision

MediaPipe Pose (BlazePose)
Real-time 3D human pose landmark detection

OpenCV
Video capture (file and webcam)
Frame preprocessing and resizing
Visualization of pose landmarks, FPS, and stance classification overlays

Mathematical Processing

NumPy - Numerical operations and data handling
Python math module - Trigonometric functions and square roots

Performance & Timing
Time module - Frame timing and FPS calculation

The current version of this project(as of 12/25/25) does not include a traditional frontend.

This project was started on 11-8-25 
