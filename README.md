# MTCNN_YOLO11
combine onet from MTCNN with yolo11 face detection wider face
```
pip install ultralytics
pip install onnx onnxruntim
```
you can use wider-face pretrained model yolov11 as face detection
then for face recognition, you need face alignment, here is the best location, so follow me
# the challenge:

mtcnn have 3 part: pnet, rnet, onet: at last step ie onet, landmark is the output of network so: 
input: face cropted result from yolo11 (trained on wider face)
model: accept 1,3,48,48 rgb format float32
so after preprocss give the input to onet, then post process the output for alignment wiht simple image processing trasformation on image. 
# how to use: 
```
python main_onnx.py #for run onet in onnx format, recommened (do not need to pytorch!)
python main.py #for run onet in torch format (same as MTCNN_pytorch release)
```
you can change the input image path in main.py to run model on different images
I am sure it is a submodule in a big project, so no need to install this project as a package , .. just use these modules 
