"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import math
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import pathlib

import zipfile

from craft import CRAFT

from collections import OrderedDict


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (1,1,150,150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,75,75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,75,75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)

        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output




def recognize_char(image, classes, checkpoint, num_classes, predicted_text):
    #classes = ["0","1", "2", "1", "4", "5", "0", "5", "2"]
    # train_path = r'E:\CLASSIFICATION_USING_CNN\train'
    # root = pathlib.Path(train_path)
    # classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    # print(type(classes))

    checkpoint = torch.load(checkpoint)
    model = ConvNet(num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.eval()

    # Transforms
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5), (0.5))
        # transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
        #                   [0.5,0.5,0.5])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    image = clahe.apply(image) + 5

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # cv2.imshow("clahe", image)

    image = Image.fromarray(image)

    # For reversing the operation:
    image = np.asarray(image)
    # image = Image.open(img_path)

    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax()

    pred = classes[index]
    predicted_text.append(pred)
    print(pred)
    return  predicted_text


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='E:\\M&M\\images_only_camera_position_fixed\\v\\u\\y', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(count, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    count+= 1
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
   # print(boxes)
    if(len(boxes)>0):

        for i in boxes:
           # print(i[0][0])
            #print(i[1][0])
            #print(i[2][0])
            #print(i[3][0])
            #points = np.array([[[i[0][0], i[0][1]], [i[1][0], i[1][0]], [i[2][0], i[2][1]], [i[3][0], i[3][1]]]])
            points = np.array([[[int(i[0][0]), int(i[0][1])], [int(i[1][0]), int(i[1][1])], [int(i[2][0]), int(i[2][1])], [int(i[3][0]), int(i[3][1])]]])
            points_2 = np.float32([[int(i[0][0]), int(i[0][1])], [int(i[1][0]), int(i[1][1])], [int(i[2][0]), int(i[2][1])], [int(i[3][0]), int(i[3][1])]])
            slope = (int(i[1][1]) - int(i[0][1]))/(int(i[1][0]) - int(i[0][0]))
            angle = math.atan(slope)
            angle = math.degrees(angle)
          #  print(angle)
      #  print(points)
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        # method 1 smooth region
        cv2.drawContours(mask, points, -1, (255, 255, 255), -1, cv2.LINE_AA)
        res2 = cv2.bitwise_and(image, image, mask=mask)
        rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
        cropped = res2[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        height_crop, width_crop = cropped.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width_crop / 2, height_crop / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rotated_image = cv2.warpAffine(src=cropped, M=rotate_matrix, dsize=(width_crop, height_crop))
        cv2.imwrite(f"C:\\Users\\sikhin.vc\\PycharmProjects\\CRAFT_test\\CRAFT-pytorch\\roi\\rotated_image_d_{count}.jpg", rotated_image)
        cv2.imshow("roi", rotated_image)
        cv2.waitKey(1)
        height_rotated = rotated_image.shape[0]
        width_rotated = rotated_image.shape[1]
        print(width_rotated, height_rotated)
        width_individual_char = int(width_rotated / 9)
        th_x, th_y = 2, 2
        initial_x, initial_y = 0+th_x, 0+th_y
        predicted_text = []

        for i in range(9):
            crop_ind_img = rotated_image[initial_y:initial_y + height_rotated+th_y, initial_x:initial_x + width_individual_char + th_x]
         #   cv2.imwrite(f'ind_img_{i}{count}.jpg', crop_ind_img)
            cv2.imshow("characters", crop_ind_img)
            alpha = [0, 1, 4]
            num = [2, 3, 5, 6, 7, 8]

            initial_x = int(initial_x + width_individual_char)
            if i in alpha:
                classes = ["A", "B", "E", "I", "J", "K", "O", "S", "Z"]
                checkpoint = ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
                checkpoint = r'C:\Users\sikhin.vc\PycharmProjects\CRAFT_test\CRAFT-pytorch\best_checkpoint_alphabets.model'
                num_classes = 9
            if i in num:
                classes = ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
                checkpoint = r'C:\Users\sikhin.vc\PycharmProjects\CRAFT_test\CRAFT-pytorch\best_checkpoint_numbers.model'
                num_classes = 9

            predicted_text = recognize_char(crop_ind_img, classes, checkpoint, num_classes, predicted_text)
            print(predicted_text)
            cv2.waitKey(1)


        cv2.imshow("rotated", rotated_image)
        cv2.waitKey(0)
        cv2.imwrite("roi.jpg", cropped)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, count


count = 0
if __name__ == '__main__':
    # load net

    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
   # for k, image_path in enumerate(image_list):
    vid = cv2.VideoCapture(r"E:\M&M\M&M recordings-20211116T050112Z-001\M_M recordings\2021-10-13_15-30-00.mp4")
    while True:

        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        #image = imgproc.loadImage(image_path)
        ret, image = vid.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow("input_frame", cv2.resize(image, (500, 400)))
        cv2.waitKey(1)
        bboxes, polys, score_text, count = test_net(count, net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
       # filename, file_ext = os.path.splitext(os.path.basename(image_path))
       # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        #cv2.imwrite(mask_file, score_text)
        #cv2.imshow("out", score_text)
       # cv2.waitKey(0)

        #file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
