#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from segmentation_srvs.srv import Segmentation
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import random
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import os
os.environ['YOLO_VERBOSE'] = 'false'
class YOLOv8HumanMaskService(Node):
    def __init__(self):
        super().__init__('yolov8_human_mask_service')
        self.srv = self.create_service(Segmentation, 'human_mask', self.handle_human_mask)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8s-seg.pt')  # Ensure this model supports segmentation
        self.human_class_id = 0  # Class ID for person class
        print('YOLOv8 model loaded successfully, node started')

    def handle_human_mask(self, request, response):
        # self.get_logger().info('Human mask request received')
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(request.req, desired_encoding='rgb8')
        # Perform human detection and segmentation using YOLOv8
        results = self.model.predict(cv_image, classes=[self.human_class_id], conf=0.7)
        # Get class IDs, bounding boxes, and masks from the results

        if results is not None:
            class_ids = results[0].boxes.cls  # class IDs for each detection
            bboxes = results[0].boxes.xyxy  # bounding boxes in xyxy format
            if(len(bboxes)>0):
                human_bboxes = bboxes[class_ids == self.human_class_id][0] # (left, top, right, bottom)
                masks = results[0].masks.xy  # segmentation masks
                human_masks = masks[self.human_class_id]
                xmin = int(human_bboxes[0])
                ymin = int(human_bboxes[1])
                xmax = int(human_bboxes[2])
                ymax = int(human_bboxes[3])

                points = np.int32([human_masks])
                cv2.fillPoly(cv_image, points, [255,0,0])
                # create a mask from the image
                mask = cv2.inRange(cv_image, np.array([255, 0, 0]), np.array([255, 0, 0]))
                bitwise_mask = cv2.bitwise_not(mask)
                # draw bbox
                to_show = cv_image.copy()
                cv2.rectangle(to_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                # cv2.imshow('mask', bitwise_mask)

                # cv2.imshow('mask', to_show)
                # cv2.waitKey(1)
                # Convert OpenCV image to ROS Image message
                # self.get_logger().info('Human mask generated')
                response.mask = self.bridge.cv2_to_imgmsg(bitwise_mask, encoding="mono8")
                response.bbox = [xmin, ymin, xmax, ymax]
                response.found_human = True
                return response
        
        # return empty mask if no person detected
        ros_image = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        response.mask = self.bridge.cv2_to_imgmsg(ros_image, encoding="mono8")
        response.bbox = [0, 0, 0, 0]
        response.found_human = False
        # self.get_logger().info('No person detected')
        # cv2.imshow('No detection', ros_image)
        # cv2.waitKey(1)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8HumanMaskService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()