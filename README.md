# RoadBoundaryFeatures  
#### Prerequisites
```console
pip install -r requirements.txt
```

DVC is configured with a remote using my ssh access on mps-yellowstone. You will have to configure a new remote before first usage.

#### Training

Training depends on two stages  

	+-------+         +------------+  
	| setup |         | checkpoint |  
	+-------+         +------------+  
		 **        **  
		   **    **  
		     *  *  
		  +-------+  
		  | train |  
		  +-------+  

Run training with

```console
dvc repro
```
or only the train stage:
```console
python train.py --args
```
#### Workflow
After training, track the changes in DVC

	git add -u
	git commit -m "message"

optionally: 
	git tag "comet_ml id"

	dvc push

#### Tags

The follwing tag have been used for experiments in the thesis:

##### Datasets
4stages-carla: 759df0806f10447b8291fddffa1903c7  
4stages-nuscenes: c66455d1fb4846f986a703479dfd3492  

##### Depth
4stages-carla: 759df0806f10447b8291fddffa1903c7  
3stages-carla: ec8c683bf0f04a5cb0d84dd504592c26  
2stages-carla: 0a8182a60b404624b97bc49b415eb6f0  

4stages-lyft: 38c50728e13d4c9fb9067c97fad8948b  
3stages-lyft: 4e43e36dac194858ab2d921c6f076a38  

##### Upsampling
4stages-carla: 759df0806f10447b8291fddffa1903c7  
4stages-carla-nearest: 023cd3038bbc434a89df13b9e6a73761  

##### Loss
4stages-carla: 759df0806f10447b8291fddffa1903c7  
4stages-carla-fixed-loss: 9e8b8c73f51d4283bd70872912d9807a  

4stages-lyft: 38c50728e13d4c9fb9067c97fad8948b  
4stages-lyft-fixed-loss: 264f9dcf41164b8cbdd5a17d54d762d0  
