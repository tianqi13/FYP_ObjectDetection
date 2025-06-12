# Object Detection with 3D Bounding Boxes and Map Generation
A YOLO-World and DepthAnythingV2 based object detection pipeline for underwater object detection. 
This project then uses YOLO-World, EfficientVitSAM and RoMA to generate a 3D map of the environment.

## Installation Steps
1. Clone this repository
```bash
git clone https://github.com/tianqi13/FYP_ObjectDetection.git
cd FYP_ObjectDetection
```
2. Install required packages 
```bash
pip install -r requirements.txt
```
Next, install **mmcv**. YOLO-World is built on mmcv, and the newest pre-built package only supports cuda 12.1 and torch 2.4.

```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

The other option is to build mmcv from source, but this takes a much longer time (not recommended).
```bash
pip install -U openmim
mim install mmcv==2.2.0
```

3. Download checkpoints

YOLO-World:
```bash
mkdir YOLO_world/weights/pre_train
wget -P YOLO_world/weights/pre_train https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage2-b3e3dc3f.pth

mkdir -p YOLO_world/weights/finetune
wget -P YOLO_world/weights/finetune https://huggingface.co/Tianqi13/FYP_ObjectDetection/resolve/main/l_finetune.pth

mkdir YOLO_world/weights/prompt_tune
wget -P YOLO_world/weights/prompt_tune https://huggingface.co/Tianqi13/FYP_ObjectDetection/resolve/main/l_prompt_tuned.pth
```

DepthAnythingV2:
```bash
mkdir -p DepthV2/checkpoints

#vits works fine, but if you want to use other configurations you can download all the weights
wget -P DepthV2/checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth 
```

EfficientViTSAM:
```bash
mkdir -p EfficientViTSAM/assets/checkpoints/efficientvit_sam

#vits works fine, but if you want to use other configurations you can download all the weights
wget -P EfficientViTSAM/assets/checkpoints/efficientvit_sam https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt
```

## Run Demo for Object Detection
The 2 python scripts for Object Detection demonstration are: 
1. run_image.py: This draws bounding boxes on an input image. The output image is labelled output_bbox_image.png
2. run_video.py: This draws bounding boxes on an input video. The output video is labelled output_bbox_video.mp4

Note: You may run into this error: AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0. You can edit the source file's 'mmcv_maximum_version' to 2.2.1. There are two source files where this needs to be changed, one in python3.12/site-packages/mmdet/__init.py__ and another in python3.12/site-packages/mmyolo/__init.py__

You can change the image/video inputs, as well as the detection prompts(class_names) and model configurations in these files. Look for this part at the top of the script:

```python
# ''' CHANGE CONFIGURATIONS IF NEEDED
path_to_image = 'img_L.png'
class_names=['bottle', 'cup', 'soda can', 'cone']
detector = ObjectDetector(model_weights='finetuned', class_names=class_names)   
depth_estimator = DepthEstimator(model_config='vits')    
score_thr = 0.65 #reduce this if you want to detect more objects, but it will also increase false positives
nms_thr = 0.5                                              
# '''
```

You can also download a sample video "test.mp4". 
```bash
wget https://huggingface.co/Tianqi13/FYP_ObjectDetection/resolve/main/test.mp4
```
## Run Demo for Map Generation
For the map generation pipeline, you can run the script `run_map_gen.py`. 

However, you will need to change the camera intrinsics, the `K.txt` file in the RoMa folder. 

You will also need to provide the left and right keyframe images for stereo matching, and also the keyframe text file that contains the image name, tx, ty, tx, (T_cw) and qx, qy, qz, qw (Quaternion matrix for R_cw) for each keyframe. 

Change the paths in the script to point to your keyframe images and text file:
'''python 
keyframe_txt = 'map_gen/keyframe_images.txt'
keyframe_L = 'map_gen/img_L_kp'
keyframe_R = 'map_gen/img_R_kp'
'''



