# CV Intern Assignment for Embody VR

## Dependencies
1. Python 3.7.6
2. dlib 19.22.0
3. opencv-python 4.5.1.48
4. numpy 1.19.5
5. imutils 0.5.4

Can install dependencies by
```
pip install -r requirements.txt
```

## Running the files

Each file is the answer to the problems on the assignment sheet. Problems 1 through 3 are answered by their respective file name while problem 4 and 5 are joined together with the final file pose_tracker.py. The file centroidTracker.py contains methods used to help track and identify the faces. An example of how to run one of the files is below
```
python pose_tracker.py
```

## Notes
1. All the programs detects faces and poses through the webcam in real time. None work on image or video files. 
2. Press 'q' to exit the program
3. Pose recognition will only be for the face closest to the center
4. For problem 3 and the pose_tracker, I decided to use dlib's 68 facial landmark since it has a pretrained model that uses these landmarks so it would be easy and quick to implement. In the future I might want to test out different facial landmarks to see how many I really need to create accurate results. 