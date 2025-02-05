#define DISPLAY_OGL 0
// ZED include
#include <sl/Camera.hpp>
// OpenCV
#include <opencv2/opencv.hpp>
// ROS2 includes
#include <rclcpp/rclcpp.hpp>
// executor
#include "GLViewer.hpp"
#include "PracticalSocket.h"
#include "rclcpp/visibility_control.hpp"
#include "segmentation_srvs/srv/segmentation.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "utils.hpp"
#include "yolov8/yolov8_seg.h"

using namespace sl;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("zed_body_publisher");
  auto point_cloud_pub =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 1);
  auto point_cloud_pub_rh_z_up =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "point_cloud_rh_z_up", 1);

  // instantiate yolo model
  RCLCPP_INFO(node->get_logger(), "Loading YOLOv8 model");
  srand((unsigned)time(NULL));
  std::string netPath = "/home/nardi/SKEL_WS/ros2_ws/yolov8s-seg.onnx";
  cv::dnn::Net net;
  Yolov8Seg yolov8Seg;
  if (!yolov8Seg.ReadModel(net, netPath, false)) {
    std::cout << "ReadModel failed" << std::endl;
    return -1;
  }
  RCLCPP_INFO(node->get_logger(), "YOLOv8 model loaded");
  // cv::Mat srcImg = cv::imread("/home/nardi/SKEL_WS/ros2_ws/2.jpg");
  // if (srcImg.empty()) {
  //   std::cout << "Read image failed" << std::endl;
  //   return -1;
  // }
  // RCLCPP_INFO(node->get_logger(), "Detecting");
  // std::vector<OutputParams> output;
  // if (!yolov8Seg.Detect(srcImg, net, output)) {
  //   std::cout << "Detect failed" << std::endl;
  //   return -1;
  // }
  // for (int i = 0; i < output.size(); ++i) {
  //   std::cout << "id: " << output[i].id
  //             << " confidence: " << output[i].confidence
  //             << " box: " << output[i].box << std::endl;
  //   cv::imshow("mask", output[i].boxMask);
  //   cv::waitKey(0);
  // }

  Camera zed;
  InitParameters init_parameters;
  init_parameters.camera_resolution = RESOLUTION::HD720;
  init_parameters.camera_fps = 30;
  init_parameters.depth_mode = DEPTH_MODE::NEURAL;
  // init_parameters.coordinate_system =
  // COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  init_parameters.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
  init_parameters.svo_real_time_mode = true;

  parseArgsMonoCam(argc, argv, init_parameters);

  auto returned_state = zed.open(init_parameters);
  if (returned_state != ERROR_CODE::SUCCESS) {
    zed.close();
    return EXIT_FAILURE;
  }

  PositionalTrackingParameters positional_tracking_parameters;
  // positional_tracking_parameters.set_floor_as_origin = true;

  returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
  if (returned_state != ERROR_CODE::SUCCESS) {
    zed.close();
    return EXIT_FAILURE;
  }

  BodyTrackingParameters body_tracker_params;
  body_tracker_params.enable_tracking = true;
  body_tracker_params.enable_body_fitting = true;
  body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;
  body_tracker_params.detection_model =
      BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
  body_tracker_params.enable_segmentation = true;
  returned_state = zed.enableBodyTracking(body_tracker_params);
  if (returned_state != ERROR_CODE::SUCCESS) {
    zed.close();
    return EXIT_FAILURE;
  }

#if DISPLAY_OGL
  GLViewer viewer;
  viewer.init(argc, argv);
#endif

  Pose cam_pose;
  cam_pose.pose_data.setIdentity();

  BodyTrackingRuntimeParameters body_tracker_parameters_rt;
  body_tracker_parameters_rt.detection_confidence_threshold = 50;
  // body_tracker_parameters_rt.minimum_keypoints_threshold = 7;

  Bodies bodies;

  unsigned int serial_num = 0;
  bool run = true;

  std::string servAddress;
  unsigned short servPort;
  UDPSocket sock;

  servAddress = "230.0.0.1";
  servPort = 20000;

  std::cout << "Sending fused data at " << servAddress << ":" << servPort
            << std::endl;

  RuntimeParameters rt_params;
  // rt_params.measure3D_reference_frame = REFERENCE_FRAME::WORLD;
  rt_params.measure3D_reference_frame = REFERENCE_FRAME::CAMERA;

  std::cout << "Sending Mono-Camera data at " << servAddress << ":" << servPort
            << std::endl;

  cv::Mat image;
  auto point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  std::vector<OutputParams> output;

  while (rclcpp::ok() && run) {
    auto err = zed.grab(rt_params);
    if (err == ERROR_CODE::SUCCESS) {
      sl::Mat sl_image;
      zed.retrieveImage(sl_image, VIEW::LEFT);

      cv::Mat cvImage(sl_image.getHeight(), sl_image.getWidth(),
                      (sl_image.getChannels() == 1) ? CV_8UC1 : CV_8UC4,
                      sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));

      if (cvImage.channels() == 4) {
        cv::Mat tmpImage;
        cv::cvtColor(cvImage, tmpImage, cv::COLOR_BGRA2BGR);
        cvImage = tmpImage;
      }
      // draw bbox
      zed.retrieveBodies(bodies, body_tracker_parameters_rt);
      if (bodies.body_list.size() == 0) {
        cv::Mat resized = cvImage.clone();
        cv::resize(resized, resized, cv::Size(), 0.5, 0.5);
        cv::imshow("video", resized);
        cv::waitKey(1);
        continue;
      }

#if DISPLAY_OGL
      viewer.updateData(bodies, cam_pose.pose_data);
#endif

      if (bodies.is_new) {
        try {
          double distance_to_camera = 1000000000000;
          int closest_body = 0;
          for (int i = 0; i < bodies.body_list.size(); i++) {
            if (bodies.body_list[i].position.z < distance_to_camera) {
              distance_to_camera = bodies.body_list[i].position.z;
              closest_body = i;
            }
          }

          // continue;
          std::string data_to_send = getJson(zed, bodies, closest_body,
                                             body_tracker_params.body_format)
                                         .dump();
          sock.sendTo(data_to_send.data(), data_to_send.size(), servAddress,
                      servPort);

          sl::Mat point_cloud;
          zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

          if (!yolov8Seg.Detect(cvImage, net, output)) {
            std::cout << "Detect failed" << std::endl;
            return -1;
          }
          if (output.size() == 0) {
            std::cout << "No objects detected" << std::endl;
            continue;
          }
          int cls_id = -1;
          for (int it = 0; it < output.size(); it++) {
            if (output[it].id == 0) {
              // detected an human
              cls_id = it;
              break;
            }
          }
          if(cls_id == -1){
            std::cout << "No human detected" << std::endl;
            continue;
          }
          cv::Rect box = output[cls_id].box;
          cv::Mat boxMask = output[cls_id].boxMask;
          int bb_x_min = box.x;
          int bb_y_min = box.y;
          int bb_x_max = box.x + box.width;
          int bb_y_max = box.y + box.height;

          auto ros_pointcloud = sensor_msgs::msg::PointCloud2();
          ros_pointcloud.header.stamp = node->now();
          ros_pointcloud.header.frame_id = "zed2_left_camera_frame";

          ros_pointcloud.width = bb_x_max - bb_x_min;
          ros_pointcloud.height = bb_y_max - bb_y_min;
          ros_pointcloud.is_dense = false;
          ros_pointcloud.is_bigendian = false;

          sensor_msgs::msg::PointField x_field, y_field, z_field, rgb_field;
          x_field.name = "x";
          x_field.offset = 0;
          x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
          x_field.count = 1;
          y_field.name = "y";
          y_field.offset = 4;
          y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
          y_field.count = 1;
          z_field.name = "z";
          z_field.offset = 8;
          z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
          z_field.count = 1;
          rgb_field.name = "rgb";
          rgb_field.offset = 12;
          rgb_field.datatype = sensor_msgs::msg::PointField::UINT32;
          rgb_field.count = 1;

          ros_pointcloud.fields.push_back(x_field);
          ros_pointcloud.fields.push_back(y_field);
          ros_pointcloud.fields.push_back(z_field);
          ros_pointcloud.fields.push_back(rgb_field);
          ros_pointcloud.point_step = 16;
          ros_pointcloud.row_step =
              ros_pointcloud.point_step * ros_pointcloud.width;
          ros_pointcloud.data.resize(ros_pointcloud.row_step *
                                     ros_pointcloud.height);
          ros_pointcloud.is_dense = false;

          auto ros_pointcloud_rh_z_up = ros_pointcloud;

          float *data = reinterpret_cast<float *>(ros_pointcloud.data.data());
          float *data_rh_z_up =
              reinterpret_cast<float *>(ros_pointcloud_rh_z_up.data.data());

          sl::float4 point3D;

          //   // use bbox to mask the point cloud
          int index = 0;
          for (int y = bb_y_min; y < bb_y_max; y++) {
            for (int x = bb_x_min; x < bb_x_max; x++) {
              if (boxMask.at<uchar>(y - bb_y_min, x - bb_x_min) == 0) {
                continue;
              } else {
                // mask original image pixel
                cv::Vec3b &pixel = cvImage.at<cv::Vec3b>(y, x);
                pixel[2] = 255;
                point_cloud.getValue(x, y, &point3D);
                if (index < ros_pointcloud.width * ros_pointcloud.height) {
                  data[index * 4 + 0] = point3D.x / 1000.0;
                  data[index * 4 + 1] = point3D.y / 1000.0;
                  data[index * 4 + 2] = point3D.z / 1000.0;

                  data_rh_z_up[index * 4 + 0] = point3D.z / 1000.0;
                  data_rh_z_up[index * 4 + 1] = -point3D.x / 1000.0;
                  data_rh_z_up[index * 4 + 2] = point3D.y / 1000.0;

                  uint32_t rgb = *reinterpret_cast<uint32_t *>(&point3D.w);
                  // convert from ABGR to RGBA
                  rgb = ((rgb & 0x000000FF) << 16) | ((rgb & 0x0000FF00)) |
                        ((rgb & 0x00FF0000) >> 16);
                  std::memcpy(&data[index * 4 + 3], &rgb, 4);
                  std::memcpy(&data_rh_z_up[index * 4 + 3], &rgb, 4);
                  index++;
                }
              }
            }
          }
          //   // draw bounding box
          cv::rectangle(cvImage, cv::Point(bb_x_min, bb_y_min),
                        cv::Point(bb_x_max, bb_y_max), cv::Scalar(255, 0, 0),
                        3);
          cv::resize(cvImage, cvImage, cv::Size(), 0.5, 0.5);
          cv::imshow("video", cvImage);
          cv::waitKey(1);
          point_cloud_pub->publish(ros_pointcloud);
          point_cloud_pub_rh_z_up->publish(ros_pointcloud_rh_z_up);
        } catch (SocketException &e) {
          std::cerr << e.what() << std::endl;
        }
      }
    } else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
      zed.setSVOPosition(0);
    } else {
    }

#if DISPLAY_OGL
    run = viewer.isAvailable();
#endif

    sl::sleep_ms(10);
  }

#if DISPLAY_OGL
  viewer.exit();
#endif

  bodies.body_list.clear();

  zed.disableBodyTracking();
  zed.disablePositionalTracking();
  zed.close();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
