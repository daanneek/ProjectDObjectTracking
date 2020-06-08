import threading
import argparse
import numpy as np
import cv2 as cv
import sys
from object_detection import object_detector


def drawPred(frame, objects_detected):
    print(objects_detected)
    for object_, info in objects_detected.items():
        box = info[0]
        confidence = info[1]
        label = '%s: %.2f' % (object_, confidence)
        point_one = (int(box[0]), int(box[1]))
        point_two = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(frame, point_one, point_two, (100, 100, 100))
        left = int(box[0])
        top = int(box[1])
        size_of_label, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, size_of_label[1])
        cv.rectangle(frame, (left, top - size_of_label[1]), (left + size_of_label[0], top + base_line), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))  

def postprocess(frame, out, threshold, classes, framework):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    objects_detected = dict()

    if framework == 'Caffe':
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > threshold:
                left = int(detection[3] * frame_width)
                top = int(detection[4] * frame_height)
                right = int(detection[5] * frame_width)
                bottom = int(detection[6] * frame_height)
                
                class_identification = int(detection[1])
                i = 0
                label = classes[class_identification]
                label_with_num = str(label) + '_' + str(i)
                while(True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = [(int(left),int(top),int(right - left), int(bottom-top)),confidence] 

    else:
        for detection in out:
            confidences = detection[5:]
            class_identification = np.argmax(confidences)
            confidence = confidences[class_identification]
            if confidence > threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = center_x - (width / 2)
                top = center_y - (height / 2)
                i = 0
                label = classes[class_identification]
                label_with_num = str(label) + '_' + str(i)
                while True:
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = [(int(left),int(top),int(width),int(height)),confidence]

    return objects_detected

def intermediate_detections(stream, predictor, threshold, classes):
    _,frame = stream.read()
    predictions = predictor.predict(frame)
    objects_detected = postprocess(frame, predictions, threshold, classes, predictor.framework)

    object_list = list(objects_detected.keys())
    print('Tracking the following objects', object_list)

    trackers_dict = dict()    

    if len(object_list) > 0:

        trackers_dict = {key : cv.TrackerKCF_create() for key in object_list}
        for item in object_list:
            trackers_dict[item].init(frame, objects_detected[item][0])

    return stream, objects_detected, object_list, trackers_dict

def process(args):
    objects_detected = dict()
    cv.setUseOptimized(onoff=True) 
    tracker = cv.TrackerKCF_create()


    predictor = object_detector(args.model, args.config)
    stream = cv.VideoCapture(args.input if args.input else 0)
    window_name = "Tracking in progress"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)        
    cv.moveWindow(window_name, 10, 10)

    if args.output:
        _, test_frame = stream.read()
        height = test_frame.shape[0]
        width = test_frame.shape[1]
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(args.output, fourcc, 20.0, (width, height))

    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    else:
        classes = list(np.arange(0,100))

    stream, objects_detected, objects_list, trackers_dict = intermediate_detections(stream, predictor, args.thr, classes)    

    while stream.isOpened():
    
        grabbed, frame = stream.read()

        if not grabbed:
            break

        timer = cv.getTickCount()        

        if len(objects_detected) > 0:
            del_items = []
            for objectt, tracker in trackers_dict.items():
                positive, bbox = tracker.update(frame)
                if positive:
                    objects_detected[objectt][0] = bbox
                else:
                    print('Failed to track ', objectt)
                    del_items.append(objectt) 
        
            for item in del_items:            
                trackers_dict.pop(item)
                objects_detected.pop(item)
                
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        if len(objects_detected) > 0:
            drawPred(frame, objects_detected)
            cv.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1.35, (50, 170, 50), 1)

        else:
            cv.putText(frame, 'Tracking Failure. Trying to detect more objects', (50, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            stream, objects_detected, objects_list, trackers_dict = intermediate_detections(stream, predictor, args.thr, classes)
            
        
        cv.imshow(window_name, frame)

        if args.output:
            out.write(frame)
        k = cv.waitKey(1) & 0xff

        #Refresh detection by pressing 'q'
        if k == ord('q'):
            print('Refreshing. Detecting New objects')
            cv.putText(frame, 'Refreshing. Detecting New objects', (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            stream, objects_detected, objects_list, trackers_dict = intermediate_detections(stream, predictor, args.thr, classes)
            
        #Save results and end program if 'e' pressed.
        if k == ord('e') : break 

    stream.release()
    if args.output:
        out.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Object Detection and Tracking on Video Streams')

    parser.add_argument('--input', help='Path to file, leave empty if webcam')

    parser.add_argument('--output', help='Path to output file location. leave empty to delete results')

    parser.add_argument('--model', required=True,
                        help='Path to a binary file of model contains trained weights. '
                             'It could be a file with extensions .caffemodel (Caffe), '
                             '.weights (Darknet)')

    parser.add_argument('--config',
                        help='Path to a text file of model contains network configuration. '
                             'It could be a file with extensions .prototxt (Caffe), .cfg (Darknet)')
    
    parser.add_argument('--classes', help='Optional path to a text file with names of classes to label detected objects.')
    
    parser.add_argument('--thr', type=float, default=0.35, help='Confidence threshold for detection')

    args = parser.parse_args()

    start_thread = threading.Thread(target=process, args=(1,))
    start_thread.start()

    process(args)

if __name__ == '__main__':
    main()
