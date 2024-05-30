# import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
import pyautogui
import win32api, win32con, win32gui
import cv2
import math
import time
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image


def draw_bounding_boxes(image_location, bounding_boxes):
    # Read the image
    image = cv2.imread(image_location)

    # Convert the bounding box tensor to numpy array
    # bounding_boxes = bounding_boxes.cpu().numpy()
    # print(bounding_boxes)
    # Iterate through each bounding box
    for box in bounding_boxes:
        # Extract the coordinates of the bounding box
        print("draw box : ", box)
        xmin, ymin, xmax, ymax = box
        # Draw the bounding box on the image
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# detector = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")

model = YOLO('best.pt')

# Display model information (optional)
model.info()


size_scale = 3

cls_done = [False,False,False,False,False,False,]

# Get rect of Window
hwnd = win32gui.FindWindow(None, 'Counter-Strike 2')
# hwnd = win32gui.FindWindow("UnrealWindow", None) # Fortnite
rect = win32gui.GetWindowRect(hwnd)
region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

while True:


    # Get image of screen
    ori_img = np.array(pyautogui.screenshot(region=region))
    ori_img = cv2.resize(ori_img, (ori_img.shape[1] // size_scale, ori_img.shape[0] // size_scale))
    image = np.expand_dims(ori_img, 0)
    img_w, img_h = image.shape[2], image.shape[1]

    # Detection
    # result = detector(image)
    # result = {key:value.numpy() for key,value in result.items()}
    # boxes = result['detection_boxes'][0]
    # scores = result['detection_scores'][0]
    # classes = result['detection_classes'][0]

    yolo_results = model.predict(source=ori_img, conf=0.25, save=False)
    boxes = yolo_results[0].boxes.xyxyn
    boxex = boxes.cpu().numpy()

    cls_list = yolo_results[0].boxes.cls


    # Check every detected object
    detected_boxes = []

    for i, box in enumerate(boxes):
        # Choose only person(class:1)
        xmin, ymin, xmax, ymax = box

        # if(cls_done[int(cls_list[i])] == False):
        #     cls_done[int(cls_list[i])] = True
        #     image_to_save = Image.fromarray(ori_img)
        #     image_to_save.save(f'class_image{int(cls_list[i])}.png')
        #     draw_bounding_boxes(f'class_image{int(cls_list[i])}.png', [ yolo_results[0].boxes.xyxy[i] ] )

        # ymin, xmin, ymax, xmax = tuple(box)
        print("detect box xmin, ymin, xmax, ymax : ")
        print(box)

        if ymin > 0.5 and ymax > 0.8: # CS:Go
        #if int(xmin * img_w * 3) < 450: # Fortnite
            continue
        left, right, top, bottom = int(xmin * img_w), int(xmax * img_w), int(ymin * img_h), int(ymax * img_h)
        detected_boxes.append((left, right, top, bottom))
        #cv2.rectangle(ori_img, (left, top), (right, bottom), (255, 255, 0), 2)

    print("Detected:", len(detected_boxes))
    print(yolo_results[0].boxes.cls)

    # Check Closest
    if len(detected_boxes) >= 1:
        min = 99999
        at = 0
        centers = []
        for i, box in enumerate(detected_boxes):
            x1, x2, y1, y2 = box
            c_x = ((x2 - x1) / 2) + x1
            c_y = ((y2 - y1) / 2) + y1
            centers.append((c_x, c_y))
            dist = math.sqrt(math.pow(img_w/2 - c_x, 2) + math.pow(img_h/2 - c_y, 2))
            if dist < min:
                min = dist
                at = i

        # Pixel difference between crosshair(center) and the closest object
        x = centers[at][0] - (img_w/2)
        y = centers[at][1] - img_h/2 - (detected_boxes[at][3] - detected_boxes[at][2]) * 0.30

        # Move mouse and shoot
        scale = 1.65 * size_scale
        x = int(x * scale )
        y = int(y * scale)

        # print("Aim : ", x, y)
        # mx, my = pyautogui.position()
        # print("before : ", mx, my)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
        # mx2, my2 = pyautogui.position()
        # print("After : ", mx2, my2)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        # time.sleep(0.08)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    #ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("ori_img", ori_img)
    #cv2.waitKey(1)

    # time.sleep(0.08)
