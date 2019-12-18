# video-labeler
Simple Python approach to manually labeling video frames to generate training data for supervised machine learning models. 


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
Haven't been tested yet. A list of required packages is provided in `environment-packages.yaml` - use this file to manually install the packages using conda/pip/whatever you like to try and get it working. 

