# Empty_training
Training code for empty page detection model.

## Intro

The model can be used for classifying input image files to 'ok' and 'empty' document.

The following input file formats are accepted: 

- .jpg
- .png 
- .tiff 

## Setup

### Creating virtual environment using Anaconda

The model can be safely installed and used in its own virtual environment. 
For example an Anaconda environment 'empty_env' can be created with the command

### Installing dependencies in LINUX (Tested with Ubuntu 20.04)

```conda create -n empty_env python=3.7```

Now the virtual environment can be activated by typing

```conda activate empty_env```

Install required dependencies/libraries by typing 

```
pip install -r requirements.txt
```

The latter command must be executed in the folder where the requirements.txt file is located.

NB! If you are having problems with the above command, installing this library can help. `conda install -c conda-forge poppler`

## Training

Before training the data should be organized into a train folder, where there are two folders: an empty folder, containing empty images, and a ok folder for non-empty images. An example of the organization is depicted below.

```
├──Empty_training
      ├──models
      ├──data
      |   ├──train
      |       ├──empty
      |       └──ok
      ├──runs
      ├──train.py
      ├──utils.py
      ├──dataset.py
      ├──constants.py
      └──requirements.txt
```

An example of training command. More arguments can be found in the train.py file or by running `python train.py -h`. 

```python train.py --epochs 10 --run_name 'run1'```

By default, the code uses a pretrained model trained on empty images found in the `./models` folder. If you want to train from scratch you can specify it in the training command as below. 

```python train.py --epochs 10 --run_name 'run1' --from_scratch True```

If you want to use data augmentations (i.e. rotate, colorjitter, sharpness, blur, affine and erasing), you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --own_transform True```

If you want to use different learning rates for base model and classification head, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --double_lr True```

If you want to use freeze the base model during training, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --freeze True```

If you want to change learning rate, number of workers or batch size, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --lr 0.01 --num_workers 4 --batch 8```

All of the examples above can be used in one command.

### Visualizing training metrics with Tensorboard

You can visualize training metrics with Tensorboard by running a following command in the training folder.

```tensorboard --logdir runs```

After this you can in your preferred browser go to this link http://localhost:6006/. There you can see how your training is progressing. 

### Information about the saved the models

The models are saved into `./runs/models` folder.

The code saves two models based on different metrics. The first model is saved based on a "fitness" score that is calculated `0.75 * recall_ok + 0.25 * balanced_accuracy`. This is because, when filtering empty images, it is important that as much of the ok images are found. The second model is saved based on validation loss.