import cv2
from ultralytics import YOLO
import numpy as np
import onnxruntime
import numpy as np
from aling_transform import warp_and_crop_face, calibrate_box, get_reference_facial_points

ort_session = onnxruntime.InferenceSession(r"D:\BagherAl\projects\MAF_ALIGNMENT\landmark_onet.onnx", 
                                           providers=['CPUExecutionProvider'])

def mtcnn_step3_onnx(image, bounding_box, threshold=0.8):
    bounding_boxes = np.array([bounding_box])
    # bounding_boxes = ndarray.shape= (1,4or5)
    onnxruntime_input = {"input":image}
    ort_outputs = ort_session.run(None, onnxruntime_input)
    # Extract the outputs
    landmarks = ort_outputs[0]  # shape [n_boxes, 10]
    offsets = ort_outputs[1]    # shape [n_boxes, 4]
    probs = ort_outputs[2]      
    if probs[:, 1] < threshold:
        print("fail to landmark")
        return None
    keep = np.where(probs[:, 1] > threshold)[0]
    bounding_boxes = bounding_boxes[keep]
    offsets = offsets[keep]
    landmarks = landmarks[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0 # x2 -x1
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0 # y2- y1
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]
    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    return bounding_boxes, landmarks


def squre_bounding_box(x1, y1, x2, y2, original_shape):
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width // 2
    cy = y1 + height // 2
    side = max(width, height)
    new_x1 = max(0, cx - side//2)
    new_y1 = max(0, cy - side//2)
    new_x2 = min(original_shape[1], new_x1 + side)
    new_y2 = min(original_shape[0], new_y1 + side)
    
    return new_x1, new_y1, new_x2, new_y2
             
def preprocess(img, size=(48, 48)):
    """Preprocessing step before feeding the network.
    Arguments:
        img: a uint numpy array of shape [h, w, c] BGR format.
    Returns:
        a float numpy array of shape [1, c, h, w] RGB format.
    """
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125
    return img


image_path = r"need_alignment.jpg"
img=cv2.imread(image_path)
model = YOLO("face_detection_yolov11m_widerface.pt")  # pretrained YOLO11n model
# Run batched inference on a list of images
results = model.predict(image_path)
conf=results[0].boxes.conf.item()
x1, y1, x2, y2 = list(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))

x1, y1, x2, y2 = squre_bounding_box(x1, y1, x2, y2, results[0].orig_shape)
face_crop = results[0].orig_img[y1:y2, x1:x2, :]
face_crop = preprocess(face_crop)
onnx_input = {"input":face_crop}
_, landmarks = mtcnn_step3_onnx(face_crop, bounding_box=(x1, y1, x2, y2,conf))
facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
img_show = results[0].orig_img
#comment in future 2 bellow lines
for point in facial5points:
    cv2.circle(img_show, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color
# Display the image with landmarks
cv2.imshow('Image with Landmarks', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
refrence = get_reference_facial_points(default_square= True)
warped_face = warp_and_crop_face(results[0].orig_img, facial5points, refrence, crop_size=(112,112))

#uncomment when vector model will use:
# warped_face = cv2.cvtColor(warped_face, cv2.COLOR_BGR2RGB)

cv2.imshow('alignmented face', warped_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("")