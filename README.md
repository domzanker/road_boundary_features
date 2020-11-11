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
