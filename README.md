[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# MIVisionX Applications

A Compilation of all MIVisionX applications available open-source

MIVisionX has several applications built on top of OpenVX and its modules, it uses AMD optimized libraries to build applications that can be used as prototypes or used as models to develop products.

## Computer Vision Applications

### Bubble Pop

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/bubble_pop) creates bubbles and donuts to pop using OpenVX & OpenCV functionality.

<p align="center"> <img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/vx-pop-app.gif"> </p>

### SkinTone Detector

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/samples#skintonedetectgdf) is set to showcase how to use AMD's OpenVX and RunVX application.

<p align="center"> <img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/samples/images/skinToneDetect_image.PNG"> </p>

### Canny Edge Detector

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/samples#cannygdf) is set to showcase how to use AMD's OpenVX and RunVX application.

<p align="center"> <img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/samples/images/canny_image.PNG"> </p>

## Computer Vision & Machine Learning Applications

### Recognize Digits

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/dg_test#mivisionx-dgtest) is used to recognize handwritten digits.

<p align="center"> <img src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/DGtest.gif"> </p>

### Cloud Application

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/cloud_inference#cloud-inference-application) does inference using a client-server system.

<p align="center"> <a href="http://www.youtube.com/watch?v=0GLmnrpMSYs"> <img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/inferenceVideo.png"> </a></p>

### MIVisionX Inference Analyzer

[MIVisionX Inference Analyzer Application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/apps/mivisionx_inference_analyzer#mivisionx-python-inference-analyzer) using pre-trained `ONNX` / `NNEF` / `Caffe` models to analyze and summarize images.

<p align="center"><img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/inference_analyzer.gif" /></p>

### Image Augmentation

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/apps/image_augmentation#image-augmentation-application) demonstrates the basic usage of RALI's C API to load JPEG images from the disk and modify them in different possible ways and displays the output images.

<p align="center"> <img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/image_augmentation.png" /> </p>

### MIVisionX OpenVX Classsification

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/apps/mivisionx_openvx_classifier/README.md) shows how to run supported pre-trained caffe models with MIVisionX RunTime.

<p align="center"> <img src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/mivisionx_openvx_classifier_imageClassification.png"></p>

### MIVisionX Validation Tool

[MIVisionX ML Model Validation Tool](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/apps/mivisionx_validation_tool#mivisionx-python-ml-model-validation-tool) using pre-trained `ONNX` / `NNEF` / `Caffe` models to analyze, summarize, & validate.

<p align="center"><img width="90%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/docs/images/validation-2.png" /></p>

### MIVisionX WinML Classification

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/apps/mivisionx_winml_classifier/README.md) shows how to run supported ONNX models with MIVisionX RunTime on Windows.

<p align="center"> <img width="60%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/apps/mivisionx_winml_classifier/images/MIVisionX-ImageClassification-WinML.png"> </p>

### MIVisionX WinML YoloV2

This sample [application](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/blob/master/apps/mivisionx_winml_yolov2#yolov2-using-amd-winml-extension) shows how to run tiny yolov2(20 classes) with MIVisionX RunTime on Windows.

<p align="center"> <img width="60%" src="https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/raw/master/apps/mivisionx_winml_yolov2/image/cat-yolo.jpg"> </p>

### Classifier

[MIVisionX-Classifier](https://github.com/kiritigowda/MIVisionX-Classifier) - This application runs know CNN image classifiers on live/pre-recorded video stream.

<p align="center"> <img width="60%" src="https://github.com/kiritigowda/MIVisionX-Classifier/raw/master/data/classifier.png"> </p>

### YoloV2

[YOLOv2](https://github.com/kiritigowda/YoloV2NCS) - Run tiny yolov2 (20 classes) with AMD's MIVisionX

<p align="center"> <img width="60%" src="https://github.com/kiritigowda/YoloV2NCS/raw/master/data/yolo_dog.jpg"> </p>

### Traffic Vision

[Traffic Vision](https://github.com/srohit0/trafficVision#traffic-vision) - This app detects cars/buses in live traffic at a phenomenal 50 frames/sec with HD resolution (1920x1080) using deep learning network Yolo-V2. The model used in the app is optimized for inferencing performance on AMD-GPUs using the MIVisionX toolkit.

<p align="center"> <img width="70%" src="https://raw.githubusercontent.com/srohit0/trafficVision/master/media/traffic_viosion.gif" /> </p>

### RGBD SLAM V2

[RGBDSLAMv2-MIVisionX](https://github.com/ICURO-AI-LAB/RGBDSLAMv2-MIVisionX) - This is an implementation of RGBDSLAM_V2 that utilizes AMD MIVisionX for feature detection and ROCm OpenCL for offloading computations to Radeon GPUs. This application is used to create 3D maps using RGB-D Cameras.

<p align="center"> <img width="70%" src="https://github.com/ICURO-AI-LAB/RGBDSLAMv2-MIVisionX/blob/master/media/rgbdslamv2_fr2desk_octomap.jpg?raw=true" /> </p>
