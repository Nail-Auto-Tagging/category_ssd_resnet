# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import cv2
import torch
# import resnet
import torchvision.models.resnet as resnet
import torch.nn as nn

args = {
    "model": "./model/export_model_008/frozen_inference_graph.pb",
    # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_015/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    # "labels":"record/classes.pbtxt" ,
    "num_classes": 1,
    "min_confidence": 0.6,
    "class_model": "../model/class_model/p_class_model_1552620432_.h5"}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

def test_HSV(frame):
    HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(HSV_frame)
    HH = cv2.equalizeHist(H)
    SH = cv2.equalizeHist(S)
    VH = cv2.equalizeHist(V)
    HSV_H = cv2.merge((HH, SH, VH))
    ret1, SH = cv2.threshold(SH, 0, 255, type=cv2.THRESH_OTSU)
    ret2, VH = cv2.threshold(VH, 0, 255, type=cv2.THRESH_OTSU)
    HSV_mask = cv2.bitwise_and(SH, VH)
    cv2.imshow("HSVbin", HSV_mask)

def find_hand_old(frame):
    img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (9, 9), 0)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (1, 1), 0)
    # cv2.imshow("YCrCb_frame_old", YCrCb_frame)
    # print(frame.shape[:2])
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 135, 97]), np.array([255, 177, 127]))#140 170 100 120
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 133, 77]), np.array([255, 173, 127])) # best enough
    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    # cv2.imshow("res_old", res)

    return img, bin_mask, res

def Crop_image(image):
    img_list = []
    model = tf.Graph()
    
    with model.as_default():
        graphDef = tf.compat.v1.GraphDef()

        with tf.compat.v2.io.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        # sess = tf.Session(graph=graphDef)
        # return graphDef, sess

    with model.as_default():
        with tf.compat.v1.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            
            (H, W) = image.shape[:2]
            # print("H,W:", (H, W))
            output = image.copy()
            img_ff, bin_mask, res = find_hand_old(image.copy())
            image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            # print("scores_shape:", scores.shape)
            for (box, score, label) in zip(boxes, scores, labels):
                # print(int(label))
                # if int(label) != 1:
                #     continue
                if score < args["min_confidence"]:
                    continue
                # scale the bounding box from the range [0, 1] to [W, H]
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)
                X_mid = startX + int(abs(endX - startX) / 2)
                Y_mid = startY + int(abs(endY - startY) / 2)
                cropped_region = output[startY:endY, startX:endX]
                resized = cv2.resize(cropped_region,dsize=(224,224),interpolation=cv2.INTER_LINEAR)
                tmp = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
                adjusted = np.zeros_like(resized)
                adjusted[:,:,0] = tmp
                adjusted[:,:,1] = tmp
                adjusted[:,:,2] = tmp
                normalized = adjusted / 255
                normalized = (normalized - 0.5)*2
                resized = np.transpose(normalized,(2,0,1))
                cur_res = torch.from_numpy(resized).float()
                img_list.append(cur_res)
            return img_list
            # vs.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(777)
if device == "cuda":
  torch.cuda.manual_seed_all(777)

# 미리 정의
conv1x1=resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock= resnet.BasicBlock

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64 

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) # 3 반복
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 4 반복
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 6 반복
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 3 반복
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1): # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

category = [
    'glitter','french','gradation','parts','marble','pattern','resin',
    'check','powder','onetone','cheek','syrup','character'
]

def get_category(model, input):
      
  model.eval()
  Sig = torch.nn.Sigmoid()
  input = torch.stack(input,dim=0)
  with torch.no_grad():
    input = input.to(device)
    with torch.autocast('cuda'):
        output_regular = Sig(model(input).float()).cpu()
                  
  output_regular = np.array(output_regular)
  output_regular = (output_regular > 0.5)
  return output_regular

img = cv2.imread('1426127455_0HGfZ1wc_P.jpg')
imgs = Crop_image(img)
resnet50 = ResNet(resnet.Bottleneck, [3, 4, 6, 3], len(category)).to(device) 
torch.load('category_model/v2/resnet50_categories_v2_30.pt')
res = get_category(resnet50, imgs)
print(res)
