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

## General Functionality
videolabeler is a module created for labeling mouse behavioral data. Current functionalities (with supported input formats in parenthesis) include:

* __batchFrameLabel__ - for single user labeling single recording (AVI)
* __multiLabelerBatchLabel__ - for multiple users labeling multiple recordings at once (TIFF)
* __relabel__ - for relabeling of fully labeled videos (TIFF, AVI)
* __windows_and_inspect__ - for random spot-checking of behavioral labels. Does not allow creation/adjustment of labels. (TIFF, AVI)

### Output

All labels are saved as .csv files, where each frame has an associated frame number, animal_id (if in multiLabeler mode) and label. 

MultiLabeler mode saves multiple labels, with one column for each unique labeler. As such, it's recommended to have a consistent labeler ID for all of your labels.
