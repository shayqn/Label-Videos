# video-labeler
Simple Python approach that allows users to manually label video frames. Motivation: creating training data for supervised machine learning models. 

## Install

### Linux and Mac
Use conda to create an environment from `environment-linux.yaml` (for Linux users) or `environment-macOS.yaml` (for Mac users):

`conda env create -f environment-linux.yaml` or  
`conda env create -f environment-macOS.yaml`

After creating the environment, you can activate it with:

`conda activate video-labeler-linux` or   
`conda activate video-labeler-macOS`

After activating the environment, you will need to enable the ipywidget jupyter extension for the `tqdm` package to work properly in jupyter labs. You can find instructions [here](https://ipywidgets.readthedocs.io/en/stable/user_install.html#installing-the-jupyterlab-extension). It should just require the following command: 

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`

### Other operating systems
Module works on Windows, but the environment has to created manually. A list of required packages is provided in `environment-packages.yaml` - you can use this file to manually inspect & install the packages using conda/pip/etc.

## Setup on Firefly
In order to run the module on a remote server such as Firefly, you need a remote access method with a GUI, such as VNC or X11-enabled SSH. 

Once you're connected to Firefly, clone the repository and install the Linux environment as instructed above.

### VNC
Windows users can download [UltraVNC Viewer](https://www.uvnc.com/), an open-source desktop sharing program. Once it's installed, connect to `firefly.inscopix.com:15`. Contact Help Desk if you're not sure of your VNC and server passwords.

Mac has a [built-in VNC server](https://til.hashrocket.com/posts/69cbe9b2c3-how-to-use-the-hidden-vnc-client-in-mac-osx). Connect to `firefly.inscopix.com:15` or `username@firefly.inscopix.com`. This method was not verified.

### X11
Uses who run into connection issues or other problems with VNC can use an SSH client such as PuTTy coupled with a X11-forwarding method such as XMing.

1. Download [XMing](https://sourceforge.net/projects/xming/) for Windows users or [XQuartz](https://www.xquartz.org/)
 for Mac users. Steps may be different for Mac users since method was not tested on Mac
2. Open XMing. This should be done every session. Bafflingly, XMing does nothing to indicate it's running, so just assume it's running after you double-clicked it.
3. Enable X11 forwarding on your SSH client. For PuTTy, you can find the option under Connection --> SSH --> X11
4. Connect to Firefly using SSH. The address is:
`username@firefly.inscopix.com`
5. Run the Jupyter notebook on remote server without a browser. Connecting to a specific port is optionally; this is only for illustration purposes:
`jupyter lab --port=9000 --no-browser`
6. Note the Jupyter token printed to the console. It should be something like:
`http://localhost:9000/?token=59ee5c413c6e816198a78caff7db008bb0410f98ab9ced52`
7. On your __local__ terminal, connect your local port to the remote one using: 
`ssh -N -f -L 8888:localhost:9000 username@firefly.inscopix.com`
8. In your local browser, type in the token from step 6 with your local port number
`http://localhost:8888/?token=59ee5c413c6e816198a78caff7db008bb0410f98ab9ced52`

## General Functionality
videolabeler is a module created for labeling mouse behavioral data. Current functionalities (with supported input formats in parenthesis) include:

* __batchFrameLabel__ - for single user labeling single recording (AVI)
* __multiLabelerBatchLabel__ - for multiple users labeling multiple recordings at once (TIFF)
* __relabel__ - for relabeling of fully labeled videos (TIFF, AVI)
* __windows_and_inspect__ - for random spot-checking of behavioral labels. Does not allow creation/adjustment of labels. (TIFF, AVI)


### Output

All labels are saved as .csv files, where each frame has an associated frame number and label.

|           |   label       | frame |
| ----------|:-------------:| -----:|
| 0         | walking       |   0   |
| 1         | walking       |   1   |


MultiLabeler mode saves multiple labels, with one column for each unique labeler. As such, it's recommended to have a consistent labeler ID for all of your labels.


|           |   animal_id   | Frame |  labeler_A |  labeler B |
| ----------|:-------------:| -----:|:----------:|:----------:|
| 0         | OM14          |   0   |walking     | explore    |
| 1         | OM14          |   1   |walking     | walking    |
