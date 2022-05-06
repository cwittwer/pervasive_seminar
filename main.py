"""
Receive world camera data from Pupil using ZMQ.
Make sure the frame publisher plugin is loaded and confugured to gray or rgb
"""

import zmq
from msgpack import unpackb, packb
import numpy as np
import cv2

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from _collections_abc import dict_items, dict_keys, dict_values
import utility

while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)
    
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_inception_v2_coco_2017_11_17'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict

def get_bounding_boxes(output_dict, min_score_thresh=0.5):
    boxes = output_dict['detection_boxes']
    return_boxes = []
    return_labels = []
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        # 
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name = category_index[output_dict['detection_classes'][i]]['name']
            class_id = output_dict['detection_classes'][i]
            return_boxes.append(boxes[i])
            return_labels.append((class_name,class_id))
            #print (class_name, " : ", boxes[i], output_dict['detection_classes'][i])
    
    return return_boxes, return_labels

def show_inference(model, frame):
    #take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    # Actual detection.
        
    output_dict = run_inference_for_single_image(model, image_np)
    
    boxes, labels = get_bounding_boxes(output_dict)
    #print(boxes, labels)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=5)
    
    return image_np, boxes, labels

def check_gaze_on_object(image, gaze_point, boxes, labels):
    height, width = image.shape[:2]
    x_gaze,y_gaze = gaze_point

    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i]

        left, right, bottom, top = (xmin * width, xmax * width,
                                  ymin * height, ymax * height)
        #print(left,";",right,";",x_gaze,";",top,";",bottom,";",y_gaze)
        if (x_gaze>left and x_gaze<right) and y_gaze>bottom and y_gaze<top:
            return labels[i][0],labels[i][1]
    
    return "No Object", 0

context = zmq.Context()
# open a req port to talk to pupil
addr = "127.0.0.1"  # remote ip or localhost
req_port = "50020"  # same as in the pupil remote gui
req = context.socket(zmq.REQ)
req.connect("tcp://{}:{}".format(addr, req_port))
# ask for the sub port
req.send_string("SUB_PORT")
sub_port = req.recv_string()


# send notification:
def notify(notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = "notify." + notification["subject"]
    payload = packb(notification, use_bin_type=True)
    req.send_string(topic, flags=zmq.SNDMORE)
    req.send(payload)
    return req.recv_string()


# open a sub port to listen to pupil
sub = context.socket(zmq.SUB)
sub.connect("tcp://{}:{}".format(addr, sub_port))

# set subscriptions to topics
# recv just pupil/gaze/notifications
sub.setsockopt_string(zmq.SUBSCRIBE, "frame.")
sub.setsockopt_string(zmq.SUBSCRIBE, 'gaze')


def recv_from_sub():
    """Recv a message with topic, payload.
    Topic is a utf-8 encoded string. Returned as unicode object.
    Payload is a msgpack serialized dict. Returned as a python dict.
    Any addional message frames will be added as a list
    in the payload dict with key: '__raw_data__' .
    """
    topic = sub.recv_string()
    payload = unpackb(sub.recv(), raw=False)
    extra_frames = []
    while sub.get(zmq.RCVMORE):
        extra_frames.append(sub.recv())
    if extra_frames:
        payload["__raw_data__"] = extra_frames
    return topic, payload


def has_new_data_available():
    # Returns True as long subscription socket has received data queued for processing
    return sub.get(zmq.EVENTS) & zmq.POLLIN


GAZE_CHANGE_THRESH = 3
STATE_NO = "No Object in Gaze"
world_counter = 0

recent_world = None
recent_eye0 = None
recent_eye1 = None
state = STATE_NO
prev_state = None
new_state = STATE_NO
gaze_counter = 0
x=0
y=0

FRAME_FORMAT = "bgr"

GAZE_CHANGE_THRESH = 5


# Set the frame format via the Network API plugin
notify({"subject": "frame_publishing.set_format", "format": FRAME_FORMAT})

try:
    while True:
        # The subscription socket receives data in the background and queues it for
        # processing. Once the queue is full, it will stop receiving data until the
        # queue is being processed. In other words, the code for processing the queue
        # needs to be faster than the incoming data.
        # e.g. we are subscribed to scene (30 Hz) and eye images (2x 120 Hz), resulting
        # in 270 images per second. Displays typically only have a refresh rate of
        # 60 Hz. As a result, we cannot draw all frames even if the network was fast
        # enough to transfer them. To avoid that the processing can keep up, we only
        # display the most recently received images *after* the queue has been emptied.
        while has_new_data_available():
            topic, msg = recv_from_sub()

            if topic.startswith("gaze"):
                gaze_coord = msg['norm_pos']
                confidence = msg['confidence']
                gaze_data = {'gaze_coord': gaze_coord, 'confidence': confidence}
                x = int(gaze_coord[0]*1280)
                y = int((1-gaze_coord[1])*720)
                #print(x,y)

            if topic.startswith("frame.") and msg["format"] != FRAME_FORMAT:
                print(
                    f"different frame format ({msg['format']}); "
                    f"skipping frame from {topic}"
                )
                continue

            if topic == "frame.world":
                recent_world = np.frombuffer(
                    msg["__raw_data__"][0], dtype=np.uint8
                ).reshape(msg["height"], msg["width"], 3)
            elif topic == "frame.eye.0":
                recent_eye0 = np.frombuffer(
                    msg["__raw_data__"][0], dtype=np.uint8
                ).reshape(msg["height"], msg["width"], 3)
            elif topic == "frame.eye.1":
                recent_eye1 = np.frombuffer(
                    msg["__raw_data__"][0], dtype=np.uint8
                ).reshape(msg["height"], msg["width"], 3)

        if (
            recent_world is not None
            and recent_eye0 is not None
            and recent_eye1 is not None
        ):

            recent_world = cv2.circle(recent_world, (x,y) ,radius=10, color=(0,0,255), thickness=-1)
            image_np, boxes, labels = show_inference(detection_model, recent_world)

            gaze_location, label_id = check_gaze_on_object(image_np,(x,y),boxes,labels)
            #print(gaze_location)
            if (new_state != gaze_location and gaze_counter == 3):
                #change state
                new_state = gaze_location
                gaze_counter = 0
            elif (new_state != gaze_location and prev_state == gaze_location):
                #update gaze counter
                gaze_counter+=1
            elif (new_state != gaze_location and prev_state != gaze_location):
                #update gaze counter
                prev_state = gaze_location
                gaze_counter+=0
            else:
                gaze_counter=0

            if(state != new_state):
                state = new_state
                print(state)
            screen_state = "Current Gaze: "+state

            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(image_np, screen_state,(0,25),font,1,(255,0,0),3)

            cv2.imshow("world", image_np)
            #cv2.imshow("eye0", recent_eye0)
            #cv2.imshow("eye1", recent_eye1)
            cv2.waitKey(1)
            pass  # here you can do calculation on the 3 most recent world, eye0 and eye1 images
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()