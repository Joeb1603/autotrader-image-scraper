import sys
#import argparse
import time
from pathlib import Path

import cv2
import torch
#import torch.backends.cudnn as cudnn
from numpy import random
import shutil
import warnings
import os
import numpy as np

path_to_yolo = 'F:\Programming\Dissertation-mk2\yolov7'
sys.path.insert(0, path_to_yolo)

# imports from yolov7 folder
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh 
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

car_path = "contains-car" # probably shouldn't have these outside the functions but it works
no_car_path = "no-car"

class NewLoadImages:  # Slightly modified and stripped down dataloader from yolov7

    def __init__(self, path, img_size=640, stride=32):

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

        p = str(Path(path).absolute())  # os-agnostic absolute path
   
        if os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
       
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images 
        self.nf = ni  # number of files
        self.mode = 'image'
        self.cap = None
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path

        img0 = remove_borders(img0)

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def __len__(self):
        return self.nf  # number of files

def reset_folder():
        try:
            shutil.rmtree(car_path)
        except OSError:
            print("error occured deleting folder")
        try:
            shutil.rmtree(no_car_path)
        except OSError:
            print("error occured deleting folder")
        
        Path(car_path).mkdir(parents=False, exist_ok=True)
        Path(no_car_path).mkdir(parents=False, exist_ok=True)

def remove_borders(input_img):  # uses code from https://stackoverflow.com/a/48399309
    
    if np.array_equal(input_img[0][0], [244, 246, 247]) and np.array_equal(input_img[-1][-1], [244, 246, 247]):
        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
        x,y,w,h = cv2.boundingRect(cnt)
        dst = input_img[y:y+h, x:x+w]
        return dst
    else:
        return input_img
    
def contains_car(input_image):

    source, weights, imgsz = input_image, "yolov7.pt", 640
    save_img = True
    webcam = False
    conf_thres = 0.75
    iou_thres = 0.45
    save_txt = False
    view_img = False
    save_img = True
    
    txt_name = "img0"
    just_return = False
    car_found = False   

    Path(car_path).mkdir(parents=False, exist_ok=True)
    Path(no_car_path).mkdir(parents=False, exist_ok=True)

   
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    # Initialize
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    dataset = NewLoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    print("\n")

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            txt_path = txt_name  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                      # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                    if save_txt:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    box_on_edge = False
                    upper_boundary = 0.995
                    lower_boundary = 0.005

                    if xywh[0] + xywh[2]/2 > upper_boundary or xywh[0] - xywh[2]/2 <lower_boundary:  # could be clever and do some abs() stuff here
                        box_on_edge = True

                    if xywh[1] + xywh[3]/2 > upper_boundary or xywh[1] - xywh[3]/2 <lower_boundary:
                        box_on_edge = True
                    
                    class_num = int(cls)
                    bb_area = xywh[2]*xywh[3]
                    if class_num == 2 and bb_area>0.1 and bb_area<0.9 and not box_on_edge:
                        car_found = True

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if just_return:
                return True if car_found else False
            
            # Save results (image with detections)
            if save_img:
                prefix = car_path if car_found else no_car_path
                save_path = f"img{len(os.listdir(car_path))}.jpg" if car_found else f"img{len(os.listdir(no_car_path))}.jpg"
                cv2.imwrite(os.path.join(prefix,save_path), im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

    return True if car_found else False


reset_folder()
pics = os.listdir('images')

for current_image in pics:
    contains_car(os.path.join('images',current_image))

#contains_car("0.png")