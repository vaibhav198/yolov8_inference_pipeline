import cv2
import os
from statistics import mode
from datetime import datetime
import time
# matplotlib.use('TkAgg')
from utils import DetectionModel
from utils.helpers import *
from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()

model = DetectionModel(os.getenv('MODEL_PATH'))
dest_folder = os.getenv("OUTPUT_FOLDER")
missed_folder = os.getenv("MISSED_IMAGES")


if os.getenv('MODE') == 'img':
    print("Running for IMG mode")
    try: 
        #get img folder name
        img_folder = os.getenv('IMAGES_PATH')
        img_files = os.listdir(img_folder)

        for img_file in img_files:
            print(f"Running for file: {img_file}")
            class_name = []
            image = cv2.imread(img_folder+'/'+img_file)
            orig_image = image.copy()
            results, class_names = model.pred(image)
            #print(f"class_names: {class_names}")
            #Plot rect
            for result in results:
                # print(">>>>>Helo>>>>")
                boxes = result.boxes
                boxs = boxes.xyxy
                # print(f"boxs: {boxs}")
                cls = boxes.cls
                cls = list(map(int, cls.tolist()))
                # print(f"cls###: {cls}")
                
            #     print(class_name)
                if len(cls)>0:
                    for c in range(len(cls)):
                        class_name.append(class_names[cls[c]])
                        #get the coord of bounding box
                        l = boxs[c].tolist()
                        x1, y1, x2, y2 = l[0],l[1],l[2],l[3]        
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

                        #add text
                        text = class_names[cls[c]]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        thickness = 4
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_x = int(x1+5)
                        text_y = int(y1+text_size[1]+5)
                        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,255,0), thickness)

                    c_name = mode(class_name)
                    #print(f"class name: {c_name}")
                    create_dir(dest_folder+'/'+c_name)
                    create_dir(dest_folder+'/'+c_name+'/orig')
                    create_dir(dest_folder+'/'+c_name+'/detection')
                    cv2.imwrite(dest_folder+'/'+c_name+'/detection'+'/'+img_file, image)
                    cv2.imwrite(dest_folder+'/'+c_name+'/orig'+'/'+img_file, orig_image)
                else:
                    cv2.imwrite(missed_folder+'/'+img_file, orig_image)

            print(f"Done for file: {img_file}")
    except Exception as e:
        print(f"Exception: {e}")
elif os.getenv('MODE') == 'video':
    try:   
        #get video file path
        video_path = os.getenv("VIDEO_PATH")
        # Read the video from specified path 
        cap = cv2.VideoCapture(video_path) 
        
        frame_rate = 10
        prev = 0
        # frame 
        currentframe = 0

        while(True):
            class_name = [] 
            time_elapsed = time.time() - prev
            ret,image = cap.read()

            if time_elapsed > 1./frame_rate:
                prev = time.time()
            
                if ret:
                    orig_image = image.copy()
                    results, class_names = model.pred(image)
                    #print(f"class_names: {class_names}")
                    #Plot rect
                    for result in results:
                        # print(">>>>>Helo>>>>")
                        boxes = result.boxes
                        boxs = boxes.xyxy
                        #print(f"boxs: {boxs}")
                        cls = boxes.cls
                        cls = list(map(int, cls.tolist()))
                        #print(f"cls###: {cls}")
                        if len(cls)>0:
                            for c in range(len(cls)):
                                class_name.append(class_names[cls[c]])
                                #get the coord of bounding box
                                l = boxs[c].tolist()
                                x1, y1, x2, y2 = l[0],l[1],l[2],l[3]        
                                cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), (0,255,0), 2)

                                #add text
                                text = class_names[cls[c]]
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1.5
                                thickness = 4
                                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                                text_x = int(x1+5)
                                text_y = int(y1+text_size[1]+5)
                                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,255,0), thickness)
                            c_name = mode(class_name)
                            create_dir(dest_folder+'/'+c_name)
                            create_dir(dest_folder+'/'+c_name+'/orig')
                            create_dir(dest_folder+'/'+c_name+'/detection')
                            d = datetime.now()
                            cv2.imwrite(dest_folder+'/'+c_name+'/detection/'+str(d)+'.jpg', image)
                            cv2.imwrite(dest_folder+'/'+c_name+'/orig/'+str(d)+'.jpg', orig_image)
                        else:
                            cv2.imwrite(missed_folder+'/'+'img_ved'+str(datetime.now())+'.jpg', orig_image)
                else:
                    break
        cap.release()
        cv2.destroyAllWindows() 
    except Exception as e:
        print(f"Exception: {e}")
elif os.getenv('MODE') == 'webcam':
    try:
        # Read the video from webcam 
        cap = cv2.VideoCapture(0) 
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        frame_rate = 10
        prev = 0
        # frame 
        currentframe = 0

        while(True):
            class_name = [] 
            time_elapsed = time.time() - prev
            ret,image = cap.read()

            if time_elapsed > 1./frame_rate:
                prev = time.time()
            
                if ret:
                    orig_image = image.copy()
                    results, class_names = model.pred(image)
                    #print(f"class_names: {class_names}")
                    #Plot rect
                    for result in results:
                        # print(">>>>>Helo>>>>")
                        boxes = result.boxes
                        boxs = boxes.xyxy
                        #print(f"boxs: {boxs}")
                        cls = boxes.cls
                        cls = list(map(int, cls.tolist()))
                        #print(f"cls###: {cls}")
                        if len(cls)>0:
                            for c in range(len(cls)):
                                class_name.append(class_names[cls[c]])
                                #get the coord of bounding box
                                l = boxs[c].tolist()
                                x1, y1, x2, y2 = l[0],l[1],l[2],l[3]        
                                cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), (0,255,0), 2)

                                #add text
                                text = class_names[cls[c]]
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1.5
                                thickness = 4
                                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                                text_x = int(x1+5)
                                text_y = int(y1+text_size[1]+5)
                                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,255,0), thickness)
                            c_name = mode(class_name)
                            create_dir(dest_folder+'/'+c_name)
                            create_dir(dest_folder+'/'+c_name+'/orig')
                            create_dir(dest_folder+'/'+c_name+'/detection')
                            d = datetime.now()
                            cv2.imwrite(dest_folder+'/'+c_name+'/detection/'+str(d)+'.jpg', image)
                            cv2.imwrite(dest_folder+'/'+c_name+'/orig/'+str(d)+'.jpg', orig_image)
                        else:
                            cv2.imwrite(missed_folder+'/'+'img_web'+str(datetime.now())+'.jpg', orig_image)
                else:
                    break
        cap.release()
        cv2.destroyAllWindows() 
    except Exception as e:
        print(f"Exception: {e}")
                

        
        
        

