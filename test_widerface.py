import argparse
import glob
import time
from pathlib import Path

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm

def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride 
    return max_size

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    #修改2 用循环处理所有关键点的padding和缩放
    num_pts = coords.shape[1] // 2  # 关键点数量
    x_indices = list(range(0, num_pts * 2, 2))
    y_indices = list(range(1, num_pts * 2, 2))
    coords[:, x_indices] -= pad[0]  # x padding
    coords[:, y_indices] -= pad[1]  # y padding
    coords[:, :num_pts * 2] /= gain
    for k in range(num_pts):
        coords[:, k * 2].clamp_(0, img0_shape[1])      # x
        coords[:, k * 2 + 1].clamp_(0, img0_shape[0])  # y
    return coords

def show_results(img, xywh, conf, landmarks, class_num, num_points=4):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    #修改2 用num_points替代硬编码的5
    for i in range(num_points):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i % len(clors)], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect(model, img0):
    #修改2 从模型获取关键点数量
    num_points = model.model[-1].num_points
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    if imgsz <= 0:                    # original size    
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres, num_points=num_points)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    #修改2 gn_lks动态生成
    gn_lks = torch.tensor(img0.shape)[[1, 0] * num_points].to(device)  # normalization gain landmarks
    cls_all = 5 + num_points * 2
    boxes = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        #修改2 用cls_all替代硬编码的15
        pred[:, 5:cls_all] = scale_coords_landmarks(img.shape[2:], pred[:, 5:cls_all], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            #修改2 用cls_all替代硬编码的15和10
            landmarks = (pred[j, 5:cls_all].view(1, num_points * 2) / gn_lks).view(-1).tolist()
            class_num = pred[j, cls_all].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2-x1, y2-y1, conf])
    return boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.02, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='../WiderFace/val/images/', type=str, help='dataset path')
    parser.add_argument('--folder_pict', default='/yolov5-face/data/widerface/val/wider_val.txt', type=str, help='folder_pict')
    opt = parser.parse_args()
    print(opt)

    # changhy : read folder_pict
    pict_folder = {}
    with open(opt.folder_pict, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('/')
            pict_folder[line[-1]] = line[-2]

    # Load model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    with torch.no_grad():
        # testing dataset
        testset_folder = opt.dataset_folder

        for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):
            if image_path.endswith('.txt'):
                continue
            img0 = cv2.imread(image_path)  # BGR
            if img0 is None:
                print(f'ignore : {image_path}')
                continue
            boxes = detect(model, img0)
            # --------------------------------------------------------------------
            image_name = os.path.basename(image_path)
            txt_name = os.path.splitext(image_name)[0] + ".txt"
            save_name = os.path.join(opt.save_folder, pict_folder[image_name], txt_name)
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                file_name = os.path.basename(save_name)[:-4] + "\n"            
                bboxs_num = str(len(boxes)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in boxes:
                    fd.write('%d %d %d %d %.03f' % (box[0], box[1], box[2], box[3], box[4] if box[4] <= 1 else 1) + '\n')
        print('done.')
