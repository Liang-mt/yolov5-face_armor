# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # 把下面两行的8，9删除
    # 修改2 用循环处理所有关键点的padding和缩放
    num_pts = coords.shape[1] // 2  # 关键点数量
    x_indices = list(range(0, num_pts * 2, 2))
    y_indices = list(range(1, num_pts * 2, 2))
    coords[:, x_indices] -= pad[0]  # x padding
    coords[:, y_indices] -= pad[1]  # y padding
    coords[:, :num_pts * 2] /= gain
    # clip_coords(coords, img0_shape)
    for k in range(num_pts):
        coords[:, k * 2].clamp_(0, img0_shape[1])  # x
        coords[:, k * 2 + 1].clamp_(0, img0_shape[0])  # y
    return coords


def show_results(img, xyxy, conf, landmarks, class_num, num_points=4):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    tf = max(tl - 1, 1)  # font thickness
    names = ['BG', 'B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BBs', 'BBb', 'RG', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RBs',
             'RBb', 'NG', 'N1', 'N2', 'N3', 'N4', 'N5', 'NO', 'NBs', 'NBb', 'PG', 'P1', 'P2', 'P3', 'P4', 'P5', 'PO',
             'PBs', 'PBb']
    label = str(conf)[:5]
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # 修改2 用num_points替代硬编码的4
    point = []
    for i in range(num_points):
        point.append(int(landmarks[2 * i]))
        point.append(int(landmarks[2 * i + 1]))
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i % len(clors)], -1)
    # 修改2 当关键点数为4时画四边形连线
    if num_points == 4:
        cv2.line(img, (point[0], point[1]), (point[4], point[5]), (0, 0, 255), 5)
        cv2.line(img, (point[2], point[3]), (point[6], point[7]), (0, 0, 255), 5)
        cv2.line(img, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 5)
        cv2.line(img, (point[4], point[5]), (point[6], point[7]), (0, 0, 255), 5)

    class_id = int(class_num)
    cls = f"cls:{names[class_id]}"
    cv2.putText(img, cls, (x1 + 50, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
        model,
        source,
        device,
        project,
        name,
        exist_ok,
        save_img,
        view_img,
        img_size=640,
        conf_thres=0.6,
        iou_thres=0.5
):
    print()
    print(img_size)
    # Load model
    imgsz = (img_size, img_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap in dataset:

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        # 修改 这部分取消
        # orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
        print(imgsz)
        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # 修改2 从模型获取关键点数量（兼容旧模型）
        det_head = model.model[-1]
        num_points = getattr(det_head, 'num_points', 4)
        cls_all = getattr(det_head, 'cls_all', 5 + num_points * 2)

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres, num_points=num_points)
        print(len(pred[0]), 'Armor' if len(pred[0]) == 1 else 'Armors')

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                # 修改 15改为13
                # det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], im0.shape).round()
                # 修改2 用cls_all替代硬编码的13
                #cls_all = 5 + num_points * 2
                det[:, 5:cls_all] = scale_coords_landmarks(img.shape[2:], det[:, 5:cls_all], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    # 修改 下面两行15改为13
                    # landmarks = det[j, 5:13].view(-1).tolist()
                    # class_num = det[j, 13].cpu().numpy()
                    # 修改2 用cls_all替代硬编码的13
                    landmarks = det[j, 5:cls_all].view(-1).tolist()
                    class_num = det[j, cls_all].cpu().numpy()

                    im0 = show_results(im0, xyxy, conf, landmarks, class_num, num_points=num_points)

            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov5s_416_500/weights/best.pt',help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='armor/0033.png', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=360, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', type=bool, default=True, help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img,
           img_size=opt.img_size, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
