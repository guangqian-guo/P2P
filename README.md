# P2P

The official code of our paper: "P2P: Transforming from Point Supervision to Explicit Visual Prompt for Object Detection and Segmentation" (IJCAI-24).



## Introduction

Point-supervised vision tasks, including detection and segmentation, aiming to learn a network that transforms from points to pseudo labels, have attracted much attention in recent years.  However, the lack of precise object size and boundary annotations in the point-supervised condition results in a large performance gap between point- and fully-supervised methods.  In this paper, we propose a novel iterative learning framework, Point to Prompt (P2P), for point-supervised object detection and segmentation, with the key insight of transforming from point supervision to explicit visual prompt of the foundation model.  The P2P is formulated as an iterative refinement process of two stages: Semantic Explicit Prompt Generation (SEPG) and Prompt Guided Spatial Refinement (PGSR).  Specifically, SEPG serves as a prompt generator for generating semantic-explicit prompts from point input via a group-based learning strategy. In the PGSR stage, prompts guide the visual foundation model to further refine the object regions, by leveraging the outstanding generalization ability of the foundation model. The two stages are iterated multiple times to improve the quality of predictions progressively. Experimental results on multiple datasets demonstrate that P2P achieves SOTA performance in both detection and segmentation tasks, further narrowing the performance gap with fully-supervised methods.

<img src="asserts/framework.png" alt="image-20240427161339686" style="zoom:80%;" />



## Getting started
Coming soon...

### Install





### Dataset





### Training





### Evaluation





## Citation

