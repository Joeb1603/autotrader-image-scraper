import sys
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import shutil
import warnings
import os

path_to_yolo = 'F:\Programming\Dissertation-mk2\yolov7'
sys.path.insert(0, path_to_yolo)

# imports from yolov7 folder
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

car_path = "contains-car" # probably shouldn't have these outside the functions but it works
no_car_path = "no-car"

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

    #delete_old_dirs = False


    #if delete_old_dirs:  
        

    Path(car_path).mkdir(parents=False, exist_ok=True)
    Path(no_car_path).mkdir(parents=False, exist_ok=True)

   
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    # Directories
    #save_dir = ""

    # Initialize
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

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
                    class_num = int(cls)
                    if class_num == 2:
                        #print(f"{class_num}, width = {xywh[2]:.2f} height = {xywh[3]:.2f}, area = {xywh[2]*xywh[3]:.2f}")
                        car_found = True

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            #print(f'\n{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS\n')

            
            if just_return:
                return True if car_found else False
            

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                prefix = car_path if car_found else no_car_path
                save_path = f"img{len(os.listdir(car_path))}.jpg" if car_found else f"img{len(os.listdir(no_car_path))}.jpg"
                cv2.imwrite(os.path.join(prefix,save_path), im0)
                #print(f"The image with the result is saved in: {save_path}\n")
                

    '''if save_txt or save_img:
        s = "" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")'''

    print(f'Done. ({time.time() - t0:.3f}s)')

    return True if car_found else False


reset_folder()

pics = os.listdir('images')

for current_image in pics:
    contains_car(os.path.join('images',current_image))


#contains_car("0.png")