import os
import cv2
import numpy as np

# import cv2 as cv2
#https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html
video=cv2.VideoWriter("VideoTest.avi", cv2.cv2.VideoWriter_fourcc('I','4','2','0'), 1, (384,288))  
#ourcc	4-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc. List of codes can be obtained at Video Codecs by FOURCC page. FFMPEG backend with MP4 container natively uses other values as fourcc code: see ObjectType, so you may receive a warning message from OpenCV about fourcc code conversion.
for jpgfile in glob.glob("imge/*.jpg"):
    img1 = cv2.imread(jpgfile)
	font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    #https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    img1 = cv2.putText(img1, jpgfile, (40, 40), font, 0.8, (255, 255, 255), 1)  # #添加文字
    video.write(img1)
    #cv2.imshow("Image", img1) 
    key=cv2.waitKey(100)
video.release()
cv2.destroyAllWindows()