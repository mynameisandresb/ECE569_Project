import cv2
import os

output_dir = "data"

# Creates directory for frames if it does not exist
if not os.path.exists(output_dir):
    print(f'Making directory {os.path.abspath(output_dir)}')
    os.mkdir(output_dir)

# Reads in the video
cap = cv2.VideoCapture('./IMG_0179.MOV')

# Captures first frame
success,image = cap.read()
count = 1

# Keep capturing frames, resizing and writing them out until no frames
while success:
    output_path = os.path.join(output_dir, "in%06d.jpg" % count)
    image = cv2.resize(image, (720,480))
    cv2.imwrite(output_path, image)
    success,image = cap.read()
    count += 1