# Martial-Arts-Pose-Detection
Using Mediapipe to run real time and video pose estimation and classification on American Tang Soo Do forms.

This project was made for my 2026 Governor's Honors Program application in computer science which required me to build and showcase a project with around 2 months time. 
I originally became interested in this idea because of me and my friend's shared love for martial arts. He happened to be a 1st degree black belt in American Tang Soo Do and had recently taken his black belt test which required the performance of various forms(kata). 

Being students and all, it was impossible for us to be at the gym 24/7 so a large portion of our time training martial arts took place at home without guidance. The lack of a coach guiding out movements made imitating specific moves and techniques very difficult to get right. On top of that, the lack of guidance could result in us building bad habits such as and not limited to dropping your guard hand, pushing punches, and not rotating through. 

As the resident computer scientist of the group, I decided to do some research on the idea of a form of motion capture that would let me extract the pose of someone performing a technique as a potential solution to our problem. During my research I happened upon something called A.I. pose estimation which looked very promising. Considering my limited resources and experience, A.I. pose estimation seemed like the solution to my problem as it was not particularly code-intensive to get started and there were a great number of resources on the internet that could help me. 

I was eventually able to develop a working prototype using Google’s MediaPipe framework, which specializes in real-time pose estimation. The system tracks a single subject, draws 3D body landmarks in real time, computes joint angles, and classifies stances from video or live camera input. Specifically, the idea is to have karate practitioners perform their kata and upload the video where the pose would then be calculated and display through a series of lines of points on the body drawn by the computer vision library OpenCv. 

The Pose estimation is performed using MediaPipe’s BlazePose pipeline, a two-stage system built on convolutional neural networks (CNNs). First, a person detector identifies the subject’s region of interest (ROI). This is essentially a bounding box around the main subject. It then estimates key points such as the midpoint of the hips and the body’s rotation and scale to define a bounding circle similar to the likes of Leonardo Davinci’s Vitruvian man. Once detected, a lightweight tracker follows the subject across frames, only re-running the heavier detector when tracking confidence drops. This design enables real-time performance while maintaining accuracy.
In order to draw the landmarks, Mediapipe uses the BlazePose model once again with its Convolution Neural Networks(CNN) to predict the location of the points. A CNN is a specialized type of neural network designed to work with images by sliding a small grid across the image, computing dot products(more on that later), and producing a map. They then layer the results to distinguish between pixels. The pose model predicts 33 body landmarks, each with (x, y, z, visibility) values. The model is able to label the positions of these landmarks by having the CNNs make inferences on patterns it is trained in(edges, textures, etc). The model outputs 3 dimensional results, with depth (z) estimated using an internal 3D body model. 

I originally wanted to combine Mediapipe's pose estimation model with its holistic model for a total of 74 unique points across the body: 21 landmarks on each hand and the remaining 33 points spread across the body at key joints. However it was found that layering both models causes a significant decrease in performance with fps dropping from ~30 to less than 12 frames a second. This lag made it incredibly difficult to run test footage through which and in order to avoid further performance issues, I only used the lighter pose model for my first minimal viable product.
All Mediapipe landmarks:

<img width="850" height="958" alt="image" src="https://github.com/user-attachments/assets/1a00d51a-f359-447b-b78f-bc0ede5cb09c" />

The project contains a fully functional pose detection and classification pipeline, alongside a real-time video processing loop capable of applying MediaPipe’s pose estimation A.I. to both prerecorded video and live camera feeds. Mediapipe's pose detection module supports three different model complexities: fast, medium, and heavy with each having its own trade off between computational cost and landmark accuracy. The fast model prioritizes speed over accuracy and is best suited for real-time applications on lower-end hardware while the heavy model focuses on landmark accuracy and is built for deep analysis on more powerful systems. In my project, I mainly used the medium model complexity as it strikes a good balance between the two and is more versatile.

The pose classification system currently supports 3 Tang Soo Do stances: June Bi(Starting stance), Chun Kul Cha Seh(Front stance), and Dwi Kubi(Back Stance). The original concept also included Neko Ashi Dachi(Cat stance) and Kee Ma Jaseh(Horse stance) but I unfortunately lacked the data to implement these stances accurately. Each stance is defined by a set of acceptable joint angle ranges, not fixed values, to allow for natural variations during movement. These ranges were derived by averaging angle measurements collected from multiple reference images of the same stance and the number sets were stored in the pose angles.txt file.

The joint angles are measured and calculated using the dot product and the law of cosines, all vector mathematics. A vector is simply direction + length and in a 3D space, a vector is a coordinate set of (x, y, z). To compute joints such as, let's say, the right knee angle (angle B), vectors are formed between the right knee (B) → right hip (A) and the right knee (B) → right ankle (C). Each vector is constructed using the difference between landmark coordinates: (x2−x1, y2−y1, z2−z1). The angle at the joint is then defined as the angle between vectors BA and BC. 
The joining angle(θ) is then calculated through the dot product formula. A dot product is a mathematical way to answer how aligned two vectors are and the equation is BA · BC = |BA| |BC| cos(θ). Rearranging the values gives us θ = acos.((BA · BC) / |BA| |BC|). As you can see this formula closely resembles the law of cosines. To find the magnitudes, you can use sqrt(x^2+y^2+z^2). Before applying the dot product formula, the normalized dot product(BA · BC) is clamped to the range [-1, 1] to prevent errors. Finally the resulting angle is converted into degrees for easier readability. This approach produces rotation-invariant and scale-independent measurements which makes it robust to camera orientation, scale, and other factors.

For stance classification, joint angles are computed for elbows, shoulders, hips, knees, and ankles using 3D vector mathematics. The classifier compares the current frame’s angle array against a series of accepted ranges and computes a confidence score. To handle left–right symmetry (e.g., mirrored stances), the system evaluates both the original and reversed angle arrays, allowing the stance to be recognized regardless of which leg is forward. Because the pose estimation over individual frames can be incredibly noisy or inconsistent, I implemented temporal stability through a five frame buffer. It works by displaying the most frequently appearing stance classification within the buffer period which reduces flickering.

Example Output: 
![alt text](<example output/ready stance classified 2.png>)
![alt text](<example output/front stance classified 1.png>)
![alt text](<example output/back stance classified 2.png>)

This project marked my first experience with Python, OpenCV, and computer vision, and required learning unfamiliar syntax, frameworks, and mathematical concepts. During the development process, I was forced to use external resources such as articles, tutorials, and an AI assistant (GPT-5.1) to support support my learning, especially for translating understood concepts into working code.I was careful not to rely too much on external help as it would negatively impact my learning but for certain situations it was unavoidable as much of this project lied outside my skillset.

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
