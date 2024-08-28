#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from segmentation_srvs.srv import Segmentation
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
#get package share directory
from ament_index_python.packages import get_package_share_directory
class TestClient(Node):
    def __init__(self):
        super().__init__('yolov8_human_mask_service')
        self.cli = self.create_client(Segmentation, 'human_mask')
        self.bridge = CvBridge()
        self.test_img = cv2.imread(get_package_share_directory('yolo_seg') + '/assets/student.jpg')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('Service available')
        self.call_service()

    def call_service(self):
        request = Segmentation.Request()
        request.req = self.bridge.cv2_to_imgmsg(self.test_img, encoding="bgr8")
        future = self.cli.call_async(request)
        future.add_done_callback(self.callback)
    
    def callback(self, future):
        try:
            response = future.result()
            cv_image = self.bridge.imgmsg_to_cv2(response.mask, desired_encoding='mono8')
            self.get_logger().info('Human mask received')
            cv2.imshow('Human Mask', cv_image)
            cv2.waitKey(0)
            # apply the mask to the original image
            masked_image = cv2.bitwise_and(self.test_img, self.test_img, mask=cv_image)
            cv2.imshow('Masked Image', masked_image)
            cv2.waitKey(0)
            rclpy.shutdown()
            exit(0)
            
            
        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e,))


  
def main(args=None):
    rclpy.init(args=args)
    node = TestClient()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()