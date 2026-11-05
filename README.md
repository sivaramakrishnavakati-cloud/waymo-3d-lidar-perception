# Waymo 3D LiDAR Perception & Visualization

An automotive perception engineering project using the Waymo Open Dataset to process, visualize, and explore 3D LiDAR point clouds and vehicle annotations.

## Project Overview

This project focuses on 3D environmental perception using multi-frame LiDAR data. It includes point-cloud processing, coordinate transformations, 3D vehicle bounding-box visualization, Bird's-Eye View (BEV) visualization, and an animated viewer.

## Demo

## 🎬 Project Demo

[![Waymo 3D LiDAR Perception Demo](https://img.youtube.com/vi/3SNZrspm05Q/maxresdefault.jpg)](https://www.youtube.com/watch?v=3SNZrspm05Q)

**▶ Click the preview to watch the complete interactive 3D LiDAR and BEV demonstration.**


## Key Features

* 3D LiDAR point-cloud visualization.
* Multi-frame animation across 100 sequential frames.
* Ground-truth 3D vehicle bounding-box visualization.
* Bird's-Eye View (BEV) visualization.
* Ego-vehicle position marker.
* LiDAR coordinate transformations using vehicle pose data.
* Interactive visualization with Plotly.

## Dataset

**Waymo Open Dataset – Perception**

The project uses a processed subset of 100 sequential frames from one Waymo segment.

Dataset components:

* LiDAR point clouds.
* Vehicle pose information.
* 3D ground-truth annotations.
* Vehicle bounding-box dimensions and orientation.

Dataset files are not included in this repository.

## Technology Stack

* Python
* NumPy
* Pandas
* Plotly
* PyTorch
* TensorFlow
* Waymo Open Dataset

## Project Structure

```text
lidar-3d-detection/
│
├── data/
│   └── waymo/
│       └── processed/
│           ├── multi_frame/
│           └── waymo_poses.csv
│
├── models/
│   └── waymo_bev_best_model.pth
│
├── outputs/
│   ├── waymo_animated_3d_viewer.html
│   └── Waymo_3D_LiDAR_Perception_Demo.mp4
│
├── waymo_animated_viewer.py
├── waymo_perception_viewer.py
├── visualize_waymo_ego_trajectory.py
├── extract_waymo_poses.py
└── README.md
```

## Current Implementation

The current visualization displays LiDAR point clouds and Waymo ground-truth vehicle annotations in 3D and BEV views.

An experimental BEV-based neural network was also trained using processed LiDAR data. Its predictions are still under evaluation and should not be interpreted as validated object detections.

**Important:** Red bounding boxes in the current demo represent Waymo ground-truth annotations, not neural-network predictions.

## Future Improvements

* Improve BEV object-detection performance.
* Implement 3D object detection and localization.
* Add multi-object tracking.
* Integrate camera and LiDAR perception.
* Develop an interactive perception dashboard.

## Objective

To develop practical skills in automotive perception, LiDAR data processing, 3D computer vision, and ADAS-related machine-learning applications.
