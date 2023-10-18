<style>
  .image-container {
    text-align: center;
  }
  .caption {
    font-style: italic;
    font-size: 14px;
  }
</style>


# GLPN: Monocular Depth Estimation 

Depth estimation task using a single image to predict depth.


## The Model
Global-Local Path Networks (GLPN) model trained on KITTI for monocular depth estimation. It was introduced in the paper Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth by Kim et al. and first released in this repository. 
The model was retrieved from Huggingface and converted to a .h5 file to be successfully parsed in TL platform. The model is evaluated on its train and validation sets

## The Data
The dataset the model was trained on is KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute). The KITTI dataset is one of the most popular datasets for use in mobile robotics and autonomous driving. It consists of hours of traffic scenarios recorded with a variety of sensor modalities, including high-resolution RGB, grayscale stereo cameras, and a 3D laser scanner. Despite its popularity, the dataset itself does not contain ground truth for semantic segmentation. Ros et al. labeled 170 training images and 46 testing images (from the visual odometry challenge) with 11 classes: building, tree, sky, car, sign, road, pedestrian, fence, pole, sidewalk, and bicyclist. 
The train-test split is based on Monocular Depth Estimation on [KITTI Eigen split unsupervised](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1) benchmark.



<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1;">
    <img src="images/ImageSample.png" alt="Image 1">
    <p class="caption">Input Image</p>
  </div>
  <div style="flex: 1;">
    <img src="images/ImageGT.png" alt="Image 2">
    <p class="caption">Depth Ground Truth (GT) </p>
  </div>
</div>


We notice that the GT is very sparse. The Kitti Depth Estimation GT is collected using a lidar sensor. Lidar sensors have a limited range and can't measure the depth of pixels that are too far away or blocked by other objects. 

# TL Insights

The platform automatically detects weaknesses and correlations of the model on specific clusters.

* For instance, the following cluster of samples was detected as samples with a higher SiLog error. 
These samples are all from the same recording date (2011_09_26_drive_0052_sync).


<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1;">
    <img src="images/low_performace_insight_1.png" alt="Image 1">
    <p class="caption">Low Performance Insight</p>
  </div>
  <div style="flex: 1;">
    <img src="images/PE_low_performace_cluster_1.png" alt="Image 2">
    <p class="caption">The Detected Cluster </p>
  </div>
</div>


* This recording metadata ('folder') we added for the analysis is shown to be significantly distinguished. The model's latent space is separated into several distinct clusters which are highly divided by the recording time.   
![PE_folder](images/PE_folder.png)
*TL Population Exploration samples are colored by the folder variable*



* We can see multiple caught insights of clusters that are under-representated with this variable correlated. 
Since each recording was separated into either a train set or a validation set the clusters are considered to be under-represented. 

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1;">
    <img src="images/under_rep_insights_1.png" alt="Image 1">
  </div>
  <div style="flex: 1;">
    <img src="images/under_rep_insights_2.png" alt="Image 2">
  </div>
  <div style="flex: 1;">
    <img src="images/under_rep_insights_3.png" alt="Image 3">
  </div>
  <div style="flex: 1;">
    <img src="images/under_rep_insights_4.png" alt="Image 4">
  </div>
</div>


* When one of such clusters has low performance, we can use TL to automatically find samples that are represented similarly 
to this weak cluster from a different recording folder and improve the model on.

![low_performace_cluster_representative_selection](images/low_performace_cluster_representative_selection.png)
Choosing representative from the low performance cluster 

* Also, we can qualitatively analyze this representative to better understand the errors. 

![low_performace_cluster_representative_prediction](images/low_performace_cluster_representative_prediction.png)
Representative Sample Analysis

* We notice an artifact in upper pixels of the model's prediction. In addition to that, the model misses the tree and the upper part of the building.
* This flaw occurs in all cluster samples. This is an important observation since the GT doesn't cover these parts of the image we wouldn't have another way to detect such a cluster. 


# TL Dashboard

From the dashboard, we instantly noticed that for images with higher mean depth or higher depth standard variation (std) the loss increased. 

![depth_mean_vs_loss](images/depth_mean_vs_loss.png)

![depth_std_vs_loss](images/depth_std_vs_loss.png)

We have built tests showing that the loss for such populations is above the average loss.

![GTdepth_tests](images/GTdepth_tests.png)

This might be due to the maximum detection range. When the ground truth has a higher depth than other far-away objects in the image which are not labeled, they may confound the model's prediction causing artifacts. Also, the depth variance might be related to possible noise in the measurement or objects with higher depth magnitude.  


  