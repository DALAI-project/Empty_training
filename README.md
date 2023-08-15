# Empty_training
Training code for empty page detection model.

## Intro

The model can be used for classifying input files to 'ok' and 'empty' document.

The following input file formats are accepted: 

- .pdf
- .jpg
- .png 
- .tiff 

If the input is a .pdf file, it is transformed into an image file before further processing.
Each page of a multipage .pdf file is reported as a separate document in the output file.

The results classified by the component is saved as a .csv file, where each row corresponds
to a single input file (with the exception of multipage .pdf files). The columns of the output
file contain the following data:

- Filename of the input file ('filename')
- Predicted class ('writing_type_class')
- Confidence of prediction

## Setup

### Creating virtual environment using Anaconda

The model can be safely installed and used in its own virtual environment. 
For example an Anaconda environment 'empty_env' can be created with the command

### Installing dependencies in LINUX (Tested with Ubuntu 20.04)

`conda create -n empty_env python=3.7`

Now the virtual environment can be activated by typing

`conda activate empty_env`

Install required dependencies/libraries by typing 

```
conda install -c conda-forge poppler
pip install -r requirements.txt
```

The latter command must be executed in the folder where the requirements.txt file is located.

## Training
An example of training command. More arguments can be found in the train.py file or by running `python train.py -h`. 

`python train.py --epochs 10 --train_data /path/to/train/folder --batch 32 --num_workers 4 --save_model_path /path/to/where/to/save/model/ --run_name 'run1'`

Before training the data should be organized into a train folder, where there are two folders: an empty folder, containing empty images, and a ok folder for non-empty images. An example of the organization is depicted below.


    -train
      --empty
        ---img0.jpg
        ---img1.jpg
        ---...
        ---imgn.jpg
      --ok
        ---img0.jpg
        ---img1.jpg
        ---...
        ---imgn.jpg
