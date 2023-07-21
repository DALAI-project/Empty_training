# Empty_training

## Training

`python train.py --epochs 10 --train_data /path/to/train/folder --batch 32 --num_workers 4 --save_model_path /path/to/where/to/save/model/ --run_name 'run1'`

train folder should be organized as followed:

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

Arguments:
--epochs(int) Number of epochs
--lr(float) Learning rate for optimizer
