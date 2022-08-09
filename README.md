## **Face Mesh Detection**

## Requirements
Library:
- MediaPipe
```
pip install mediapipe
```
- OpenCv
```
pip install opencv-python
```
- numpy
```
pip install numpy
```
- os
- json
- argparse
Facial landmark recognition for 68 face landmark points:
- we download from `https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat`
- we save path `./Face Mesh Detection Example/shape_predictor_68_face_landmarks.dat`
## Face Mesh Detection Basic
Face detection and Face mesh detection:
- Running following command to recognite face in image:
```
py face_detection.py
```
- And running following command to get face mesh in image:
```
py face_mesh_detection.py
```
- Running following command to help:
```
py face_detection.py -h
```
```
py face_mesh_detection.py -h
```
- The image is in `Photos` folder and face detection results will be saved in `FaceDetectionSavedImage` and face mesh results will be saved in `FaceMeshDetectionSavedImage`. We can change image and save dir:
```
usage: face_detection.py [-h] [--sav SAV] [-dir DIR_IMAGE]

face detection

options:
  -h, --help            show this help message and exit
  --sav SAV             save dir
  -dir DIR_IMAGE, --dir-image DIR_IMAGE
                        folder of image
```
```
usage: face_mesh_detection.py [-h] [--sav SAV] [-dir DIR_IMAGE]

face detection

options:
  -h, --help            show this help message and exit
  --sav SAV             save dir
  -dir DIR_IMAGE, --dir-image DIR_IMAGE
                        folder of image
```
## Face Detection Example
Delaunay Triangulation:
- Running following command with `save_dir` (default: `Delaunay triangle`) and `source_image` (default: `man.jpg`):
```
py delaunay_triangulation.py [-h] [--sav SAV] [-src SOURCE_IMAGE]
```
Face swap 684 face landmark points:
- Running following command with `save_dir` (default: `face_swapped`) and `source_image` (default: `long.jpg`), `destination_image` (default: `man_2.jpg`):
```
py face_swap_684_landmark_points.py [-h] [--sav SAV] [-src SOURCE_IMAGE] [-dst DESTINATION_IMAGE]
```
Face swap 68 face landmark points:
- Running following command with `save_dir` (default: `face_swapped`) and `source_image` (default: `long.jpg`), `destination_image` (default: `man_2.jpg`):
```
py face_swapping.py [-h] [--sav SAV] [-src SOURCE_IMAGE] [-dst DESTINATION_IMAGE]
```
