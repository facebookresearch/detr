import os
import cv2

video_path = "/home/lei/Downloads/la_4k_drive.mp4"

save_dir = "/home/lei/data/la4k_video_to_imgs"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
cap = cv2.VideoCapture(video_path)


counter = 5000
limit = counter
cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
ret, img = cap.read()
# cv2.namedWindow("test", 0)
while ret:
    if counter > limit + 2000:
        break
    # cv2.imshow("test", img)
    filename = os.path.join(save_dir, "la4k" + str(counter).zfill(8) + ".jpg")
    cv2.imwrite(filename, img)
    counter += 1
    ret, img = cap.read()