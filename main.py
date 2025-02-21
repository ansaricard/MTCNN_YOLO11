import cv2
from PIL import Image
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from aling_transform import warp_and_crop_face, calibrate_box, get_reference_facial_points

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """
        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class ONet(nn.Module):

    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load('onet.npy', allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim = -1)
        return c, b, a



def mtcnn_step3(img_boxes, bounding_box, model, threshold=0.8):
    bounding_boxes = np.array([bounding_box])
    # bounding_boxes = ndarray.shape= (1,4or5)

    output = model(img_boxes)
    landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
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
    # keep = nms(bounding_boxes, nms_threshold, mode='min')
    # bounding_boxes = bounding_boxes[keep]
    # landmarks = landmarks[keep]

    return bounding_boxes, landmarks

def mtcnn_step3_onnx(img_boxes, bounding_box, model, threshold=0.8):
    bounding_boxes = np.array([bounding_box])
    # bounding_boxes = ndarray.shape= (1,4or5)
    import onnxruntime
    onnxruntime_input = {"input":img_boxes}
    output = model(img_boxes)
    landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
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
    # keep = nms(bounding_boxes, nms_threshold, mode='min')
    # bounding_boxes = bounding_boxes[keep]
    # landmarks = landmarks[keep]

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

from ultralytics import YOLO
image_path = r"need_alignment.jpg"
img=cv2.imread(image_path)
# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onet = ONet().to(device)
onet.eval()
# Load a model
model = YOLO("face_detection_yolov11m_widerface.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model.predict(image_path)
# results[0].plot()
conf=results[0].boxes.conf.item()
x1, y1, x2, y2 = list(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))
# face_crop = results[0].orig_img[y1:y2, x1:x2, :]
x1, y1, x2, y2 = squre_bounding_box(x1, y1, x2, y2, results[0].orig_shape)
face_crop = results[0].orig_img[y1:y2, x1:x2, :]

face_crop = preprocess(face_crop)

img_boxes = torch.FloatTensor(face_crop).to(device)
_, landmarks = mtcnn_step3(img_boxes, model=onet, bounding_box=(x1, y1, x2, y2,conf))
facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
img_show = results[0].orig_img
for point in facial5points:
    cv2.circle(img_show, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green color

# Display the image with landmarks
cv2.imshow('Image with Landmarks', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
refrence = get_reference_facial_points(default_square= True)
warped_face = warp_and_crop_face(results[0].orig_img, facial5points, refrence, crop_size=(112,112))
warped_face = cv2.cvtColor(warped_face, cv2.COLOR_BGR2RGB)

# out = Image.fromarray(warped_face)
print("")