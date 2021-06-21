######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import six.moves.urllib as urllib
import numpy as np
import tarfile
import tensorflow as tf
import sys
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util



# Grab path to current working directory


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.

PATH_TO_CKPT = "path_to/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

# Path to label map file

PATH_TO_LABELS = "path_to/mscoco_label_map.pbtxt"

# Path to video
#PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
PATH_TO_VIDEO = 'path_to_video'

# Number of classes the object detector can identify
NUM_CLASSES = 1

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'path_to/test_images/'


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(category_index)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
frame = 0
while(video.isOpened()):
    ret,image_np = video.read()
    image_np = cv2.resize(image_np,(1280,720))
    frame = frame+1
    if ret and frame>90:
        frame= 0
# for item in os.listdir(PATH_TO_TEST_IMAGES_DIR):
#     if os.path.isfile(PATH_TO_TEST_IMAGES_DIR+item):
#         image = Image.open(PATH_TO_TEST_IMAGES_DIR+item)
#         #image_np = cv2.imread(PATH_TO_TEST_IMAGES_DIR+item)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        boxes = np.squeeze(output_dict['detection_boxes'])
        scores = np.squeeze(output_dict['detection_scores'])
        classes = np.squeeze(output_dict['detection_classes'])
        indices = np.argwhere(classes == 1)
        boxes = np.squeeze(boxes[indices])
        scores = np.squeeze(scores[indices])
        classes = np.squeeze(classes[indices])

    
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image_np,
        #     output_dict['detection_boxes'],
        #     output_dict['detection_classes'],
        #     output_dict['detection_scores'],
        #     category_index,
        #     instance_masks=output_dict.get('detection_masks'),
        #     use_normalized_coordinates=True,
        #     line_thickness=4)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
        
        
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)


        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.putText(image_np, f'Person: {len(boxes)}',(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Object detector', image_np)
        #cv2.waitKey(0)

    # Press 'q' to quit
    if cv2.waitKey(25) == ord('q'):
        break

# # Clean up
video.release()
cv2.destroyAllWindows()
