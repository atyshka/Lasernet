# LaserNet in Tensorflow 2.0

In this project I've done my best to implement the work from Uber ATG on LaserNet. LaserNet is a 3D object detector for autonomous driving, with 3 distinguishing factors: range-view, low-latency, and probabilistic detection. See [here](https://arxiv.org/pdf/1903.08701.pdf) for the original paper. The original paper uses Uber's proprietary dataset, and overfits on the small KITTI dataset, so for this project I am using Waymo Open Dataset.

## Instructions:

Run save_ds.py to save the dataset to disk from the cloud bucket and perform initial preprocessing
Run shard_ds.py to split the dataset into shards. This is not strictly necessary but improves results by performing an out-of-memory shuffle
Run train.py to train on the dataset

## What works:

Basic implementation of the architecture for vehicles, pedestrians, and cyclists

## What's still not done:

- Parameterize the scripts (yeah I was lazy and used hard-coded paths)
- Mean-shift cluster (coming soon)
- Adaptive NMS
- Multi-modal distriutions for vehicles 
- Integration with Waymo Dataset evaluation metrics (also coming soon)

## Example Results

![top_image_2](https://user-images.githubusercontent.com/19317207/121823595-862f2f80-cc63-11eb-9609-bda96c55dd5c.png)
