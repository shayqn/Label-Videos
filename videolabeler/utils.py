import cv2
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tnrange
from skvideo.io import FFmpegWriter
import numpy as np
import pandas as pd
import os
import random
import math

from itertools import groupby
from operator import itemgetter

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
####### Interpolation GUI functions #################
#####################################################

def interp_annotate(frames, tent_label, prev_labels):
    '''
    INPUTS:
    frames: all frames where interpolation mode is active
    tent_label <str>: tentative label
    prev_labels <list of str>: previous labels that were in the interpolate zones
    
    OUTPUTS:
    updated frames where everything has grey tent_label and green borders
    '''
    
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]
    annotated_frames = []
    
    for ind, frame in enumerate(frames):
        cv2.rectangle(frame,(0,frame_height),(frame_width,frame_height-50),(0,255,0),-1)
        cv2.rectangle(frame,(0,50),(frame_width,0),(0,255,0),-1)
        cv2.rectangle(frame,(0,frame_height),(50,0),(0,255,0),-1)
        cv2.rectangle(frame,(frame_width-50,frame_height),(frame_width,0),(0,255,0),-1)

        cv2.putText(frame,tent_label,(0,frame_height-15),cv2.FONT_HERSHEY_COMPLEX,1,(160,160,160),2,cv2.LINE_AA)
        if prev_labels[ind] != '0.0':
            cv2.putText(frame,prev_labels[ind],(500,frame_height-15),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)

        annotated_frames.append(frame)
    
    return annotated_frames


def interp_close(frames, updated_labels):
    '''
    INPUTS:
    frames: all frames with green borders and tentative labels that we want gone
    updated_labels <list of str>: previous labels that were in the interpolate zones
    
    OUTPUTS:
    frames where all frames have white borders and frames until stop_index have black tent_label
    '''
    
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]
    annotated_frames = []
    
    for ind, frame in enumerate(frames):
        cv2.rectangle(frame,(0,frame_height),(frame_width,frame_height-50),(255,255,255),-1)
        cv2.rectangle(frame,(0,50),(frame_width,0),(255,255,255),-1)
        cv2.rectangle(frame,(0,frame_height),(50,0),(255,255,255),-1)
        cv2.rectangle(frame,(frame_width-50,frame_height),(frame_width,0),(255,255,255),-1)

        if updated_labels[ind] != '0.0':
            cv2.putText(frame,updated_labels[ind],(0,frame_height-15),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2,cv2.LINE_AA)

        annotated_frames.append(frame)
    
    return annotated_frames
        

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

def PlayAndLabelFrames(frames,label_dict = {'w':'walking','t':'turning','s':'standing'}, overlap_labels=[],
                        return_labeled_frames=False,labels = []):
    
    frames_out = frames.copy()
    frame_height = frames_out[0].shape[0]
    frame_width = frames_out[0].shape[1]
    
    bordersize=50
    
    '''
    Set up variables
    '''
    #create numpy array to store the labels. Initialize as strings of zeros
    if len(labels) == 0:
        labels = np.zeros(len(frames)).astype('str')
    
    #write old overlap labels to the list
    n_overlap_labels = len(overlap_labels)
    if n_overlap_labels is not 0:
        for ind in range(0, n_overlap_labels):
            labels[ind] = overlap_labels[ind]

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
    
    interp_mode = False
    tent_label_ind = None
    
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
        Check to see if the user is using interpolation
        '''
        if key == ord('i'):
            
            #opening interpolate mode
            if interp_mode == False:
                #find last label
                nonzero_labels=[ind for ind, label in enumerate(labels[:frame_counter]) if label != '0.0']
                tent_label_ind = nonzero_labels[len(nonzero_labels)-1]
                tent_label = labels[tent_label_ind]
                
                #update GUI
                frames_out[tent_label_ind:] = interp_annotate(frames[tent_label_ind:], tent_label, labels[tent_label_ind:])
                
                #interp mode is activated
                interp_mode = True
                
            #closing interpolate mode
            else:
                '''
                Current implementation of interpolate assumes strictly chronological labeling due to the way the last label is found. 
                You can't interpolate backwards
                Once you used interpolate, you can't use it for frames before where you last used interpolate
                '''
                if frame_counter < tent_label_ind:
                    continue
                
                #find tentative label
                tent_label = labels[tent_label_ind]
                
                #update frames, including the current one
                labels[tent_label_ind:frame_counter+1] = [tent_label]*(frame_counter-tent_label_ind+1)
                
                #update GUI
                frames_out[tent_label_ind:] = interp_close(frames[tent_label_ind:], labels[tent_label_ind:])
                
                #interp mode is turned off
                interp_mode = False
                tent_label_ind = None
                
        
        elif key in label_ords:
            #don't do anything if interpolate mode is active
            if interp_mode == True: 
                continue
            
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
            cv2.rectangle(frame,(0,frame_height),(300,frame_height-50),(255,255,255),-1)
            #...the labels can be overwritten
            cv2.putText(frame,label,(0,frame_height-15),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            
            
            #update the frame (with annotation)
            frames_out[frame_counter] = frame
            #update the label array with current label
            labels[frame_counter] = label
            

        
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
            cv2.rectangle(frame,(0,frame_height),(300,frame_height-50),(255, 255, 255),-1)
            cv2.putText(frame,'no_label',(0,frame_height-15),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
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
    print(frame_height)

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
        #cv2.rectangle(frame,(0,frame_height),(300,frame_height-50),(0,0,0),-1) #solid black background
        #label text

        if label != '0.0':
            cv2.putText(frame,label,(0,985),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
            
            #overwrite the frame
            frames_out[i] = frame
    
    return frames_out


def double_annotate(frames,labels1, labels2):
    
    frames_out = frames.copy()
    
    num_frames = len(frames)
    num_labels1 = len(labels1)
    num_labels2 = len(labels2)
    
    #position the box at the lower left corner
    box_width = 250
    box_height = 75
    frame_width = frames[0].shape[0]
    frame_height = frames[0].shape[1]

    start_xy_1 = (0,frame_height)
    end_xy_1 = (box_width,frame_height-box_height)


    #assert num_frames == num_labels,'number of frames must equal number of labels
    num_to_label = min([num_frames, num_labels1, num_labels2])
    

    for i in range(num_frames):
        
        frame = frames_out[i].copy()
        label1 = labels1[i]
        label2 = labels2[i]
        
        
        '''
        for 1024x1280
        #annotate the frame with the label text
        cv2.rectangle(frame,(0,1024),(250,950),(0,0,0),-1) #need a solid background so that...
        #...the labels can be overwritten
        cv2.putText(frame,label,(0,1000),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        '''
        #annotate the frame with the label text
        #cv2.rectangle(frame,(0,900),(300,800),(0,0,0),-1) #solid black background
        #cv2.rectangle(frame,(800,900),(1100,800),(0,0,0),-1)
        #label text

        if label1 != '0.0':
            cv2.putText(frame,label,(0,985),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
            
        if label2 != '0.0':
            cv2.putText(frame,label2,(800,985),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
            
            #overwrite the frame
            frames_out[i] = frame
    
    return frames_out


def annotate_with_consensus(frames, original_labels, consensus_labels):
    
    frames_out = frames.copy()
    
    num_frames = len(frames)
    num_labels = len(original_labels)
    
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
        label = original_labels[i]
        con_label = consensus_labels[i]
        
        
        '''
        for 1024x1280
        #annotate the frame with the label text
        cv2.rectangle(frame,(0,1024),(250,950),(0,0,0),-1) #need a solid background so that...
        #...the labels can be overwritten
        cv2.putText(frame,label,(0,1000),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        '''
        #annotate the frame with the label text
        cv2.rectangle(frame,(0,900),(300,800),(0,0,0),-1) #solid black background
        cv2.rectangle(frame,(0,100),(300,0),(0,0,0),-1) #solid black background
        #label text

        if label != '0.0':
            cv2.putText(frame,label,(0,985),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)
            
        if con_label != '0.0':
            cv2.putText(frame,con_label,(0,85),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2,cv2.LINE_AA)    
        
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
########## Image Loading  ###################
#####################################################

#load batch of tiff images, given location, start and size of desired batch
def loadTiffBatch(video_dir, start, size):
    bordersize = 50
    
    batch = []
    
    for i in range(start, start + size):
        filename = os.path.join(video_dir, 'frame' + str(i) + '.tiff')
        img = cv2.imread(filename)
        border = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value = [255, 255, 255]
        )
        batch.append(border)
    
    return batch
    
    
#####################################################
########## Batch Label Video #### ###################
#####################################################

def batchFrameLabel(video_file,labels_file,batch_size,n_overlap_frames=10,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    bordersize = 50
    
    labeler_name = input('Labeler name: ')
    
    #set start_frame to 0 and initiailize master_labels
    start_frame = 0
    master_labels = pd.DataFrame()
    overlap_labels = []

    #overwrite start_frame & master_labels if user wants to continue labeling from existing label file
    if os.path.exists(labels_file):
        continue_label_input = input('Continue labeling? (N will overwrite existing {} file) [y/N]: '.format(labels_file))
        
        if continue_label_input == 'y':
            master_labels = pd.read_csv(labels_file,index_col=0)
            last_label = int(master_labels.frame.values[-1])
            start_frame = last_label - n_overlap_frames + 1
            overlap_labels = master_labels["label"].tolist()[last_label - n_overlap_frames:last_label]
            print('Loaded in {}'.format(video_file))
            print('{} frames already labeled'.format(start_frame))
            

    #load in video
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #print(n_frames)
    batch_starts = np.arange(start_frame,n_frames,batch_size-n_overlap_frames)
    #print(batch_starts.shape)
    
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
            #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            border = cv2.copyMakeBorder(
                frame,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value = [16777215, 16777215, 16777215]
            )
            frames.append(border)
            key = cv2.waitKey(1)

        # Label Frames
        label_list = PlayAndLabelFrames(frames,label_dict=label_dict,overlap_labels=overlap_labels,return_labeled_frames=False)

        #label_list = interpolate_labels(label_list) #obsolete
               
        label_df = pd.DataFrame(data = {'label':label_list,'frame':np.arange(current_batch,current_batch + n_frames_to_read,1), 'labeler': [labeler_name]*batch_size})

        #Check for save
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
            if len(overlap_labels) is not 0: #overwrite old labels
                master_labels = master_labels[:-n_overlap_frames]
            master_labels = master_labels.append(label_df)
            master_labels.to_csv(labels_file, float_format='%g')

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


#####################################################
########## Multisession batch labeling ##############
#####################################################

def multiLabelerBatchLabel(video_dir,labels_file,batch_size=500,n_overlap_frames=25,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    bordersize = 50
    
    labeler_name = input('Labeler name: ')
    
    #set start_frame to 0 and initiailize master_labels
    start_frame = 0
    master_labels = pd.DataFrame()
    overlap_labels = []

    #overwrite start_frame & master_labels if user wants to continue labeling from existing label file
    if os.path.exists(labels_file):
        master_labels = pd.read_csv(labels_file,index_col=0)
        last_label = int(master_labels.frame.values[-1])
        start_frame = last_label - n_overlap_frames + 1
        if master_labels.iloc[-n_overlap_frames].label != 'claimed':
            overlap_labels = master_labels["label"].tolist()[last_label - n_overlap_frames:last_label]
        #print('Loaded in {}'.format(video_file))
        print('{} frames already labeled'.format(start_frame))
            

    #load in video
    n_frames = len([i for i in os.listdir(video_dir) if os.path.splitext(i)[1] == '.tiff'])
    
    #print(n_frames)
    batch_starts = np.arange(start_frame,n_frames,batch_size-n_overlap_frames)
    #print(batch_starts.shape)
    
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
    claim_labeling = False
    
    while label_in_progress is True:
        #find if there are any batches claimed by user but not labeled
        claimed_df = pd.DataFrame()
        
        if os.path.exists(labels_file):
            claimed_df = master_labels.loc[(master_labels.label=='claimed') & (master_labels.labeler == labeler_name)].copy()
        
        #if the user has claimed a batch (by opening one up and not saving it), they have to label that before moving on
        if claimed_df.shape[0] != 0:
            current_batch = claimed_df.frame[0]
            claim_labeling = True
        else:
            current_batch = batch_starts[current_ind]
            claim_labeling = False
        
        # Read in video batch
        if current_batch == batch_starts[-1]:
            n_frames_to_read = n_frames - current_batch
        else:
            n_frames_to_read = batch_size
        
        #if user is not working on a previously claimed batch, claim the batch 
        if claimed_df.shape[0] == 0:
            placeholder_df = pd.DataFrame(data = {'label':['claimed']*n_frames_to_read,'frame':np.arange(current_batch,current_batch + n_frames_to_read,1), 'labeler': [labeler_name]*n_frames_to_read})
            if len(overlap_labels) is not 0: #overwrite old labels
                master_labels = master_labels[:-n_overlap_frames]
            master_labels = master_labels.append(placeholder_df)
            master_labels.to_csv(labels_file, float_format='%g')

        #load frames    
        frames = loadTiffBatch(video_dir, current_batch, n_frames_to_read)

        # Label Frames
        label_list = PlayAndLabelFrames(frames,label_dict=label_dict,overlap_labels=overlap_labels,return_labeled_frames=False)

        #label_list = interpolate_labels(label_list) #obsolete
               
        #label_df = pd.DataFrame(data = {'label':label_list,'frame':np.arange(current_batch,current_batch + n_frames_to_read,1), 'labeler': [labeler_name]*batch_size})

        #Check for save
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
            master_labels = pd.read_csv(labels_file,index_col=0)
            master_labels.loc[(master_labels.label=='claimed') & (master_labels.labeler == labeler_name), ['label']] = label_list
            master_labels.to_csv(labels_file, float_format='%g')

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
                if claim_labeling == False: 
                    current_ind += 1
                overlap_labels = label_list[batch_size-n_overlap_frames:]
            elif label_next_input == 'n':
                label_in_progress = False
            else:
                print('Input not understood, defaulting to "yes"')
                current_ind += 1


def multiLabelerBatchLabelTest(video_dir,labels_file,batch_size=500,n_overlap_frames=25,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    bordersize = 50
    min_gap = 200
    
    labeler_name = input('Labeler name: ')
    
    #set start_frame to 0 and initiailize master_labels
    master_labels = pd.DataFrame()
    frames_labeled = []
    overlap_labels = []
    
    #load in video
    n_frames = len([i for i in os.listdir(video_dir) if os.path.splitext(i)[1] == '.tiff'])

    #overwrite start_frame & master_labels if user wants to continue labeling from existing label file
    if os.path.exists(labels_file):
        master_labels = pd.read_csv(labels_file,index_col=0)   
        frames_labeled = master_labels.frame.values
        frames_left = [i for i in range(n_frames) if i not in frames_labeled]
        frames_left = np.arange(frames_left[0]-n_overlap_frames, frames_left[0]).tolist() + frames_left
    else:
        frames_left = [i for i in range(n_frames)]
        
    #generate list of contiguous frames 
    gaps = [[s, e-n_overlap_frames] for s, e in zip(frames_left, frames_left[1:]) if s+1 < e]
    edges = iter([frames_left[0]] + sum(gaps, []) + [frames_left[-1]])
    frame_ranges = list(zip(edges, edges))
    batch_starts = []
    
    #turn the list of contiguous frames into batch starts
    for frange in frame_ranges:
        n_subsections = math.ceil((frange[1] - frange[0])/(batch_size-n_overlap_frames))
        
        #if contiguous frames are long enough (ie 1000 frames with batch size 500), divide it up
        if n_subsections > 1:
            for i in range(n_subsections):
                batch_starts.append(frange[0]+i*(batch_size-n_overlap_frames))
            
            #enforce minimum gap by adjusting last batch_start if needed
            remainder = frange[1] - batch_starts[-1] + 1
            print(remainder)
            if remainder < min_gap:
                batch_starts[-1] = batch_starts[-1] - (min_gap - remainder)
                
        #else, if the contiguous frames are short, just add that to batch_starts
        else:
            batch_starts.append(frange[0])
    print(batch_starts)

    if len(batch_starts) == 0:
        print('No more unlabeled frames')
        return

    #print('Loaded in {}'.format(video_file))
    print('{} frames already labeled'.format(len(frames_labeled)))
            
    
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
    
    while label_in_progress is True:
        
        #choose random batch
        current_batch = random.choice(batch_starts)
        if os.path.exists(labels_file):
            overlap_labels = master_labels.loc[(master_labels.frame >= current_batch) & 
                                               (master_labels.frame < current_batch + n_overlap_frames)]

        # Read in video batch
        if current_batch == batch_starts[-1]:
            n_frames_to_read = n_frames - current_batch
        else:
            n_frames_to_read = batch_size

        #load frames    
        frames = loadTiffBatch(video_dir, current_batch, n_frames_to_read)

        # Label Frames
        label_list = PlayAndLabelFrames(frames,label_dict=label_dict,overlap_labels=overlap_labels,return_labeled_frames=False)
               
        label_df = pd.DataFrame(data = {'label':label_list,'frame':np.arange(current_batch,current_batch + n_frames_to_read,1), 'labeler': [labeler_name]*batch_size})

        #Check for save
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
            master_labels = pd.read_csv(labels_file,index_col=0)
            master_labels = master_labels.append(label_df)
            master_labels.to_csv(labels_file, float_format='%g')

            #print progress
            n_labeled = master_labels.shape[0]
            per_labeled = n_labeled*100 / n_frames
            print('{} out of {} frames labeled ({:.02f} %)'.format(n_labeled,n_frames,per_labeled))
            batch_starts.remove(current_batch)

        # quit if there's nothing more, continue otherwise
        if len(batch_starts) == 0:
            break


        #If user does not save, check if they want to relabel, quit or move on
        if save_labels is False:            
            cont_input = input('Continue to label? "n" for no, "r" for relabel current batch, or "c" for continue to next batch [n/r/c]: ')

            if cont_input == 'n':
                label_in_progress = False
            elif cont_input == 'r':
                pass
            elif cont_input == 'c':
                pass
            else:
                print('Input not understood. Opening same batch for relabeling.')

        else: #otherwise, just ask if they want to label the next batch

            label_next_input = input('Label next batch? [y/n]: ')

            if label_next_input == 'y':
                pass
            elif label_next_input == 'n':
                label_in_progress = False
            else:
                print('Input not understood, defaulting to "yes"')
                pass               
                
#####################################################
########## Relabel Video (in batches) #### ###################
#####################################################

def relabelFrames(video_file,labels_file,batch_size,n_overlap_frames=10,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    bordersize = 50
    labeler_name = input('Labeler name: ')
    
    #get start frame from user
    start_frame_input = input('What frame do you want to start relabeling? [enter an integer]: ')
    start_frame = int(start_frame_input)

    #read in labels
    labels = pd.read_csv(labels_file,index_col=0)

    
    #load in video
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    #batch_starts = np.arange(start_frame,n_frames,batch_size-n_overlap_frames)
    
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
    
    while label_in_progress is True:
        
        if start_frame < n_overlap_frames:
            batch_start_frame = 0
        else:
            batch_start_frame = start_frame - n_overlap_frames

        video.set(cv2.CAP_PROP_POS_FRAMES,batch_start_frame)

        #Read in video batch
        if batch_start_frame + batch_size > n_frames:
            n_frames_to_read = int(n_frames - batch_start_frame)
            end_labeling = True
        else:
            n_frames_to_read = batch_size
            end_labeling = False
        
        frames = []
        for i in tqdm(range(n_frames_to_read)):
            ret, frame = video.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            border = cv2.copyMakeBorder(
                gray,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value = [16777215, 16777215, 16777215]
            )
            frames.append(border)
            key = cv2.waitKey(1)

        #annotate frames with previous labels
        batch_labels = labels[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size))].label.values
        

        labeled_frames = annotate_frames(frames,batch_labels)
        
        # Label Frames
        label_list = PlayAndLabelFrames(labeled_frames,label_dict=label_dict,return_labeled_frames=False,labels=batch_labels)

        #label_list = interpolate_labels(label_list) #interpolate

        #Check for save 
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
            
            labels.loc[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size)),'label'] = label_list
            labels.loc[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size)),'labeler'] = [labeler_name]*batch_size
            labels.to_csv(labels_file)

        #quit if there's nothing else    
        if end_labeling is True:
            break


        #If user does not save, check if they want to relabel, quit or move on
        if save_labels is False:
            cont_input = input('Continue to label? "n" for no, "r" for relabel current batch, or "c" for continue to next batch [n/r/c]: ')

            if cont_input == 'n':
                label_in_progress = False
            elif cont_input == 'r':
                pass
            elif cont_input == 'c':
                start_frame += batch_size
            else:
                print('Input not understood. Opening same batch for relabeling.')

        else: #otherwise, just ask if they want to label the next batch

            label_next_input = input('Label next batch? [y/n]: ')

            if label_next_input == 'y':
                start_frame += batch_size
            elif label_next_input == 'n':
                label_in_progress = False
            else:
                print('Input not understood, defaulting to "yes"')
                start_frame += batch_size

                
def relabelTiff(video_dir,labels_file,batch_size,n_overlap_frames=10,
                    label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    labeler_name = input('Labeler name: ')
    
    #get start frame from user
    start_frame_input = input('What frame do you want to start relabeling? [enter an integer]: ')
    start_frame = int(start_frame_input)

    #read in labels
    labels = pd.read_csv(labels_file,index_col=0)
    
    #video = load_tiff(video_dir)
    n_frames = len([i for i in os.listdir(video_dir) if os.path.splitext(i)[1] == '.tiff'])

    #batch_starts = np.arange(start_frame,n_frames,batch_size-n_overlap_frames)
    
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
    
    while label_in_progress is True:
        
        if start_frame < n_overlap_frames:
            batch_start_frame = 0
        else:
            batch_start_frame = start_frame - n_overlap_frames

        #Read in video batch
        if batch_start_frame + batch_size > n_frames:
            n_frames_to_read = n_frames - batch_start_frame
            end_labeling = True
        else:
            n_frames_to_read = batch_size
            end_labeling = False
            
        #load frames    
        frames = loadTiffBatch(video_dir, batch_start_frame, batch_size)

        #annotate frames with previous labels
        batch_labels = labels[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size))].label.values
        

        labeled_frames = annotate_frames(frames,batch_labels)
        
        # Label Frames
        label_list = PlayAndLabelFrames(labeled_frames,label_dict=label_dict,return_labeled_frames=False,labels=batch_labels)

        #label_list = interpolate_labels(label_list) #interpolate

        #Check for save 
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
            
            labels.loc[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size)),'label'] = label_list
            labels.loc[((labels.frame >= batch_start_frame) &
                                (labels.frame < batch_start_frame + batch_size)),'labeler'] = [labeler_name]*batch_size
            labels.to_csv(labels_file)

        # quit if there's nothing more, continue otherwise
        if start_frame > n_frames - batch_size:
            break


        #If user does not save, check if they want to relabel, quit or move on
        if save_labels is False:
            cont_input = input('Continue to label? "n" for no, "r" for relabel current batch, or "c" for continue to next batch [n/r/c]: ')

            if cont_input == 'n':
                label_in_progress = False
            elif cont_input == 'r':
                pass
            elif cont_input == 'c':
                start_frame += batch_size
            else:
                print('Input not understood. Opening same batch for relabeling.')

        else: #otherwise, just ask if they want to label the next batch

            label_next_input = input('Label next batch? [y/n]: ')

            if label_next_input == 'y':
                start_frame += batch_size
            elif label_next_input == 'n':
                label_in_progress = False
            else:
                print('Input not understood, defaulting to "yes"')
                start_frame += batch_size
                
#####################################################
########## Double View Mode #### ###################
#####################################################                

                
def double_view(video_file,labels_file1,labels_file2,batch_size, label_dict = {'i':'INTERP','s':'still','r':'rearing','w':'walking', 'q':'left turn', 'e':'right turn', 'a':'left turn [still]', 'd': 'right turn [still]', 'g':'grooming','m':'eating', 't':'explore', 'l':'leap'}):
    
    '''
    This will check to see if a labels_file already exists. If so, you can choose to continue from 
    where you left off, or choose to overwrite. 
    '''
    bordersize = 50
    
    #get start frame from user
    start_frame_input = input('What frame do you want to start relabeling? [enter an integer]: ')
    start_frame = int(start_frame_input)

    #read in labels
    labels1 = pd.read_csv(labels_file1,index_col=0)
    labels2 = pd.read_csv(labels_file2,index_col=0)
    
    #load in video
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    label_in_progress = True
    
    while label_in_progress is True:       
        
        video.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

        # Read in video batch
        if start_frame + batch_size > n_frames:
            n_frames_to_read = int(n_frames - start_frame)
            end_labeling = True
        else:
            n_frames_to_read = batch_size
            end_labeling = False
        
        frames = []
        for i in tqdm(range(n_frames_to_read)):
            ret, frame = video.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            border = cv2.copyMakeBorder(
                gray,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value = [16777215, 16777215, 16777215]
            )
            frames.append(border)
            key = cv2.waitKey(1)

        #annotate frames with previous labels
        batch_labels1 = labels1[((labels1.frame >= start_frame) &
                                (labels1.frame < start_frame + batch_size))].label.values

        batch_labels2 = labels2[((labels2.frame >= start_frame) &
                                (labels2.frame < start_frame + batch_size))].label.values
        
        labeled_frames = double_annotate(frames,batch_labels1, batch_labels2)
        
        # Label Frames
        label_list = PlayAndLabelFrames(labeled_frames,label_dict=label_dict,return_labeled_frames=False,labels=batch_labels1)   

        label_next_input = input('See next batch? [y/n]: ')

        if label_next_input == 'y':
            start_frame += batch_size
        elif label_next_input == 'n':
            label_in_progress = False
        else:
            print('Input not understood, defaulting to "yes"')
            start_frame += batch_size
            
            
#####################################################
########## Window and Inspect #### ###################
#####################################################                

#given a list of labels, return most common label and consensus 
def smooth_labels(labels):
    common_label = max(set(labels), key=labels.count)
    consensus =labels.count(common_label)/len(labels)
    
    return common_label, consensus

                
def window_and_inspect(video_file, label_file, window_size=10, overlap_size=3,start_frame=None):
    
    bordersize = 50
    video = cv2.VideoCapture(video_file)
    
    #read in labels
    df = pd.read_csv(label_file,index_col=0)
    labels = df["label"].to_list()
    
    
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #if start_frame isn't specified, choose it randomely.
    if start_frame is None:    
        start_frame = random.randint(0, n_frames - window_size)
    
    print('Frames: ' + str(start_frame) + ' to ' + str(start_frame+window_size))
    
    #Read in video batch
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []

    for i in tqdm(range(window_size)):
        ret, frame = video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        border = cv2.copyMakeBorder(
                gray,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value = [16777215, 16777215, 16777215]
            )
        frames.append(border)
        key = cv2.waitKey(1)

    #annotate frames with label + consensus
    raw_labels = labels[start_frame:start_frame + window_size]
    
    label, consensus = smooth_labels(raw_labels)
    
    con_label = label + ' (' + str(consensus) + ')'
    
    labeled_frames = annotate_with_consensus(frames, raw_labels, [con_label]*window_size)
    
    # Label Frames
    PlayAndLabelFrames(labeled_frames)

    
    
def window_and_inspect_tiff(video_dir, label_file, window_size=10, overlap_size=3,start_frame=None):
    
    #read in labels
    df = pd.read_csv(label_file,index_col=0)
    labels = df["label"].to_list()   
    
    n_frames = len([i for i in os.listdir(video_dir) if os.path.splitext(i)[1] == '.tiff'])
    
    #if start_frame isn't specified, choose it randomely.
    if start_frame is None:    
        start_frame = random.randint(0, n_frames - window_size)
    
    print('Frames: ' + str(start_frame) + ' to ' + str(start_frame+window_size))
    
    #Read in video batch
    
    frames = loadTiffBatch(video_dir, start_frame, window_size)

    #annotate frames with label + consensus
    raw_labels = labels[start_frame:start_frame + window_size]
    
    label, consensus = smooth_labels(raw_labels)
    
    con_label = label + ' (' + str(consensus) + ')'
    
    labeled_frames = annotate_with_consensus(frames, raw_labels, [con_label]*window_size)
    
    # Label Frames
    PlayAndLabelFrames(labeled_frames)    
   