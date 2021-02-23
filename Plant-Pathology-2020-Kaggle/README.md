# Plant Pathology

## Run

1. `git clone` this repo.
2. Prepare image dataset and put the images into `data/sample_img/`
3. goto `code/`, then  
    `$ python3 main.py` for training.  
    `$ python3 GUI.py` to start GUI and start evaluation and visualization.  
    <div align="left">
    <img src="./ReadmeImages/initial_screenshot.png" height="300px" alt="initial" >
    <img src="./ReadmeImages/pick_screenshot.png" height="300px" alt="pick image" >
    <img src="./ReadmeImages/predict_screenshot.png" height="300px" alt="predict" >
    </div>
    <!-- ![initial](ReadmeImages/initial_screenshot.png)
    ![pick](ReadmeImages/pick_screenshot.png)
    ![predict](ReadmeImages/predict_screenshot.png) -->

## Attention Maps Visualization  

Code in `code/eval.py` helps generate attention maps. (croped image, Heat attention map, Image x Attention map)  

<div align="center">
<img src="./ReadmeImages/raw.jpg" height="200px" alt="Raw" >
<img src="./ReadmeImages/heat_atten.jpg" height="200px" alt="Heat" >
<img src="./ReadmeImages/raw_atten.jpg" height="200px" alt="Atten" >
</div>

<!-- ![](./ReadmeImages/raw.jpg)  
![](./ReadmeImages/heat_atten.jpg)
![](./ReadmeImages/raw_atten.jpg) -->

## Loss graph  

<div align="center">
<img src="./ReadmeImages/train_figure.png" height="500px" alt="loss accuracy graph" >
</div>