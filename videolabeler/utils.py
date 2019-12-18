import cv2
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tnrange
import numpy as np

#####################################################
####### Load Video Frames ##########################
#####################################################
def LoadVideoFrames(video_file,num_frames=None):
    video = cv2.VideoCapture(video_file)
    frames = []
    
    if num_frames is None:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(num_frames),desc='Loading video'):
        # Read video capture
        ret, frame = video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        frames.append(gray)

        key = cv2.waitKey(1)

    video.release()
    
    return frames

#####################################################
####### Play Video Frames ###########################
#####################################################
def PlayVideoFrames(frames):
    
    playVideo = True

    frame_counter = 0
    while playVideo is True:

        frame = frames[frame_counter]
        cv2.imshow('video',frame)

        key = cv2.waitKey(0)

        while key not in [ord('q'),ord(','),ord('.')]:
            key = cv2.waitKey(0)

        if key == ord('.'):
            frame_counter += 1
        elif key == ord(','):
            frame_counter -= 1
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#####################################################
####### Play & Label Video Frames ###################
#####################################################
def on_trackbar(val):
    return

def setFrameCounter(frame_counter,num_frames):
    
    #if the user has scrolled past the end, go to the beginning 
    if frame_counter == num_frames:
        frame_counter = 0 
    #if user has scrolled to the left of the beginning, go to the end
    elif frame_counter == -1:
        frame_counter = num_frames -1
    
    return frame_counter

def PlayAndLabelFrames(frames,label_dict = {'i':'INTERP','w':'walking','t':'turning','s':'standing'},
                        return_labeled_frames=False):
    
    frames_out = frames.copy()
    '''
    Set up variables
    '''
    #create numpy array to store the labels. Initialize as strings of zeros
    labels = np.zeros(len(frames)).astype('str')

    #get the key ords and names for each label
    label_ords = [ord(k) for k in list(label_dict.keys())]
    label_names = list(label_dict.values())
    #create a dictionary that maps the key ords to the label names
    #i.e. replacing keystrokes with key ords as the dict keys
    label_key_dict = {}
    for label_ord,label_name in zip(label_ords,label_names):
        label_key_dict[label_ord] = label_name
    
    #get number of frames
    num_frames = len(frames)
    
    #initialize frame_counter, set PlayVideo boolean to True, and start displaying video
    #for labeling
    playVideo = True
    frame_counter = 0

    # create display window
    cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video',800,800)
    cv2.createTrackbar('frame', 'Video', 0,num_frames,on_trackbar)
    
    '''
    Play & Label Video
    '''
    while playVideo is True:

        #get current frame & display it
        frame_counter = cv2.getTrackbarPos('frame','Video')
        frame = frames_out[frame_counter].copy()
        cv2.imshow('Video',frame)

        #wait for keypress
        key = cv2.waitKey(0)

        '''
        Check to see if the user pressed any of the label keys
        '''
        if key in label_ords:
            #get the label name
            label = label_key_dict[key]
            
            '''
            #annotate the frame with the label text
            cv2.rectangle(frame,(0,1024),(250,950),(0,0,0),-1) #need a solid background so that...
            #...the labels can be overwritten
            cv2.putText(frame,label,(0,1000),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            '''
            
            #annotate the frame with the label text
            cv2.rectangle(frame,(0,900),(250,800),(0,0,0),-1) #need a solid background so that...
            #...the labels can be overwritten
            cv2.putText(frame,label,(0,875),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            
            #update the frame (with annotation)
            frames_out[frame_counter] = frame
            #update the label array with current label
            labels[frame_counter] = label

            '''
        Now check to see if the user to trying to control the playback
        '''
        elif key == ord(','): # if `<` then go back
            frame_counter -= 1
            frame_counter = setFrameCounter(frame_counter,num_frames)
            cv2.setTrackbarPos("frame","Video", frame_counter)

        elif key == ord('.'): # if `>` then advance
            frame_counter += 1
            frame_counter = setFrameCounter(frame_counter,num_frames)
            cv2.setTrackbarPos("frame","Video", frame_counter)

        elif key == ord('q'): #if `q` then quit
            playVideo = False


    #close any opencv windows    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    #if return_labeled_frames is True, return them along with the labels. Else just returns labels
    if return_labeled_frames:
        return labels,frames_out
    else:
        #return labels
        return labels
    

#####################################################
############# Interpolate Labels ####################
#####################################################

def interpolate_labels(labels):
    '''
    Interpolate frame labels for unlabeled frames where the previous labeled frame and the next 
    labeled frame have the same label. Interpolation is called out explictly by the user with 
    the key stroke 'i' (which will label the frame as INTERP) in the following manner:
    ['walking','INTERP','0','0','walking','0','walking']

    This function looks for frames labeled INTERP, and then applies the label from the frame
    immediately before to all the frames up until the next label. The next label must be 
    identical to the previous one, or else the interpolation fails. 

    The output of the above labels after interpolation would be:
    ['walking','walking','walking','walking','0','walking']
    '''

    interp_frames = np.where(labels == 'INTERP')[0]
    labeled_frames_all = np.where(labels != '0.0')[0]
    labeled_frames = list(set(labeled_frames_all).difference(set(interp_frames)))
    labeled_frames.sort()

    labels_interp = labels.copy()

    for interp_frame in interp_frames:

        label = labels[interp_frame-1]

        #check to see if next label is the same. It should be
        next_labeled_frame = labeled_frames[np.where(labeled_frames > interp_frame)[0][0]]
        next_label = labels[next_labeled_frame]

        assert label == next_label,'Interpolation failed because labels do not match'

        labels_interp[interp_frame:next_labeled_frame] = label
        
    return labels_interp


#####################################################
########## Annotate Frames with Labels ##############
#####################################################
def annotate_frames(frames,labels):
    
    frames_out = frames.copy()
    
    num_frames = len(frames)
    num_labels = labels.shape[0]
    
    assert num_frames == num_labels,'number of frames must equal number of labels'
    

    for i in range(num_frames):
        
        frame = frames_out[i]
        label = labels[i]
        
        if label is not '0.0':
            '''
            for 1024x1280
            #annotate the frame with the label text
            cv2.rectangle(frame,(0,1024),(250,950),(0,0,0),-1) #need a solid background so that...
            #...the labels can be overwritten
            cv2.putText(frame,label,(0,1000),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            '''
            #annotate the frame with the label text
            cv2.rectangle(frame,(0,900),(250,800),(0,0,0),-1) #need a solid background so that...
            #...the labels can be overwritten
            cv2.putText(frame,label,(0,875),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            
            #overwrite the frame
            frames_out[i] = frame
    
    return frames_out