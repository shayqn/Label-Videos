import cv2
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tnrange
from skvideo.io import FFmpegWriter
import numpy as np
import pandas as pd
import os

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

def PlayAndLabelFrames(frames,label_dict = {'i':'INTERP','w':'walking','t':'turning','s':'standing'}, overlap_labels=[],
                        return_labeled_frames=False):
    
    frames_out = frames.copy()
    frame_height = frames_out[0].shape[0]
    frame_width = frames_out[0].shape[1]
    
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
    #cv2.resizeWindow('Video',frame_width,frame_height)
    cv2.createTrackbar('frame', 'Video', 0,num_frames,on_trackbar)
    
    #show previous labels if they exist    
    n_overlap_frames = len(overlap_labels)
    
    if n_overlap_frames is not 0:
        frames_out[:n_overlap_frames] = annotate_frames(frames[:n_overlap_frames], overlap_labels)
    
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
            #cv2.rectangle(frame,(0,900),(250,800),(0,0,0),-1) #need a solid background so that...
            cv2.rectangle(frame,(0,frame_height),(300,frame_height-100),(0,0,0),-1)
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

        elif key == ord('x'): #if `x` then quit
            playVideo = False
        
        elif key == ord('\b'):
            labels[frame_counter] = 0.0
            
            #update rectangle to show label is gone
            cv2.rectangle(frame,(0,frame_height),(300,frame_height-100),(0,0,0),-1)
            cv2.putText(frame,'no_label',(0,875),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            frames_out[frame_counter] = frame


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

        label = str(labels[interp_frame-1])

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
    num_labels = len(labels)
    
    #position the box at the lower left corner
    box_width = 250
    box_height = 75
    frame_width = frames[0].shape[0]
    frame_height = frames[0].shape[1]

    start_xy = (0,frame_height)
    end_xy = (box_width,frame_height-box_height)


    assert num_frames == num_labels,'number of frames must equal number of labels'
    

    for i in range(num_frames):
        
        frame = frames_out[i].copy()
        label = labels[i]
        
        
        '''
        for 1024x1280
        #annotate the frame with the label text
        cv2.rectangle(frame,(0,1024),(250,950),(0,0,0),-1) #need a solid background so that...
        #...the labels can be overwritten
        cv2.putText(frame,label,(0,1000),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        '''
        #annotate the frame with the label text
        cv2.rectangle(frame,(0,900),(250,800),(0,0,0),-1) #solid black background
        #label text

        if label != '0.0':
            cv2.putText(frame,label,(0,875),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            
            
            #overwrite the frame
            frames_out[i] = frame
    
    return frames_out


#####################################################
########## Write annotated video  ###################
#####################################################
def writeAnnotatedVideo(write_file,annotated_frames,fps):

    video = FFmpegWriter(write_file, 
            inputdict={'-r': str(fps)},outputdict={'-r': str(fps)})

    frames = np.array(annotated_frames)

    for frame_num in tqdm(np.arange(frames.shape[0])):
        video.writeFrame(frames[frame_num,:,:])

    video.close()


#####################################################
########## Batch Label Video #### ###################
#####################################################

def batchFrameLabel(video_file,labels_file,batch_size,n_overlap_frames=10,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    
    #set start_frame to 0 and initiailize master_labels
    start_frame = 0
    master_labels = pd.DataFrame()

    #overwrite start_frame & master_labels if user wants to continue labeling from existing label file
    if os.path.exists(labels_file):
        continue_label_input = input('Continue labeling? (N will overwrite existing {} file) [y/N]: '.format(labels_file))
        
        if continue_label_input == 'y':
            master_labels = pd.read_csv(labels_file,index_col=0)
            start_frame = master_labels.frame.values[-1] + 1
            print('Loaded in {}'.format(video_file))
            print('{} frames already labeled'.format(start_frame))
            
             
    
    #load in video
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #print(n_frames)
    batch_starts = np.arange(start_frame,n_frames,batch_size-n_overlap_frames)
    #print(batch_starts.shape)
    
    #warn user if they overwrote a navigation key
    #nav_keys = [',', '.', 'f', 'q']

    #if bool(label_dict.keys() & nav_keys) is True:
    #    print("Warning: One of the navigation keys is overwritten. Do not use backspace, x, <, > as a labeling key.")

    #print all keys    
    print(""" 
    Navigation
    < : previous frame
    > : next frame
    Backspace : delete label
    x : quit
        """)

    print('Labels')
    for key in label_dict:
        print(key + " : " + label_dict[key])
            
    
    label_in_progress = True
    current_ind = 0
    overlap_labels = []
    
    while label_in_progress is True:
        
        current_batch = batch_starts[current_ind]
        
        video.set(cv2.CAP_PROP_POS_FRAMES,current_batch)

        # Read in video batch
        if current_batch == batch_starts[-1]:
            n_frames_to_read = n_frames - current_batch
        else:
            n_frames_to_read = batch_size
        frames = []

        for i in tqdm(range(n_frames_to_read)):
            ret, frame = video.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            key = cv2.waitKey(1)

        # Label Frames
        label_list = PlayAndLabelFrames(frames,label_dict=label_dict,overlap_labels=overlap_labels,return_labeled_frames=False)

        label_list = interpolate_labels(label_list) #interpolate

        label_df = pd.DataFrame(data = {'label':label_list,'frame':np.arange(current_batch,current_batch + n_frames_to_read,1)})

        save_labels_input = input('Save labels? [y/n]: ')

        if save_labels_input == 'y':
            save_labels = True
        elif save_labels_input == 'n':
            save_labels = False
        else:
            print('Input not understood, defaulting to yes')
            save_labels = True

        #Save labels
        if save_labels is True:
            master_labels = master_labels.append(label_df)
            master_labels.to_csv(labels_file)

            #print progress
            n_labeled = master_labels.shape[0]
            per_labeled = n_labeled*100 / n_frames
            print('{} out of {} frames labeled ({:.02f} %)'.format(n_labeled,n_frames,per_labeled))

        # quit if there's nothing more, continue otherwise
        if current_batch == batch_starts[-1]:
            break


        #If user does not save, check if they want to relabel, quit or move on
        if save_labels is False:
            cont_input = input('Continue to label? "n" for no, "r" for relabel current batch, or "c" for continue to next batch [n/r/c]: ')

            if cont_input == 'n':
                label_in_progress = False
            elif cont_input == 'r':
                pass
            elif cont_input == 'c':
                current_ind += 1
                overlap_labels = label_list[batch_size-n_overlap_frames:]
            else:
                print('Input not understood. Opening same batch for relabeling.')

        else: #otherwise, just ask if they want to label the next batch

            label_next_input = input('Label next batch? [y/n]: ')

            if label_next_input == 'y':
                current_ind += 1
                overlap_labels = label_list[batch_size-n_overlap_frames:]
            elif label_next_input == 'n':
                label_in_progress = False
            else:
                print('Input not understood, defaulting to "yes"')
                current_ind += 1