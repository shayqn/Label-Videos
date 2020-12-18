import sys
import os

import cv2

starttime = 0 # minutes
stoptime = 0 # minutes
fps = 20 # default fps

startframe = int(starttime * 60 * fps)
stopframe = int(stoptime * 60 * fps ) 

def avi_to_tiff(vid_file, save_dir, startframe = 0, stopframe=0):
    video = cv2.VideoCapture(vid_file)

    if stopframe == 0:
        stopframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print('\nReading {}'.format(os.path.basename(vid_file)))
    print('\tConverting {} to {} frames of movie...'.format(startframe, stopframe))
    count = startframe
    for i in (range(startframe,stopframe)):
        video.set(1, i)
        frame = video.read()[1]
        #ret, frame = video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_dir + "frame"+str(count)+".tiff", gray)
        count += 1

    video.release()


stop_ex = 0
if len(sys.argv)>1:
    vid_file = sys.argv[1]
    save_dir = sys.argv[2]
else:
    print('Enter path to movie file and path to tiffs folder')
    stop_ex = 1


if (os.path.isdir(save_dir) == 0) or (os.path.isfile(vid_file) ==0 ):
    print('Enter valid file and/or directory')
    stop_ex = 1
    
if stop_ex == 0:
    avi_to_tiff(vid_file, save_dir, startframe = startframe, stopframe = stopframe)        
