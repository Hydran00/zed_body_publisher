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
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "utils.hpp"
#include "yolov8/yolov8_seg.h"
#include <cv_bridge/cv_bridge.h>

using namespace sl;
using namespace std::chrono_literals;

void show_resized_img(cv::Mat &img, float scale, std::string name) {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(), scale, scale);
  cv::imshow(name, resized);
  cv::waitKey(1);
}

void show_resized_fhd(cv::Mat &img, std::string name) {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(1920, 1080)); // Resize to Full HD
  cv::imshow(name, resized);
  cv::waitKey(1);
}
void publish_image(
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub,
    const cv::Mat &image, const std::string &frame_id) {
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock().now();
  header.frame_id = frame_id;
  // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Publishing image with %d
  // channels",
  //             image.channels());
  // cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
  auto image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  image_pub->publish(*image_msg);
}

int main(int argc, char **argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("zed_body_publisher");

  // instantiate yolo model
  RCLCPP_INFO(node->get_logger(), "Loading YOLOv8 model");
  srand((unsigned)time(NULL));

  // declare parameters
  node->declare_parameter("suffix", "");
  node->declare_parameter("point_cloud_topic_name", "point_cloud");
  node->declare_parameter("full_point_cloud_topic_name", "");
  node->declare_parameter("image_topic_name", "camera_raw");
  node->declare_parameter("camera_stream", true);
  node->declare_parameter("resolution", "2K");
  node->declare_parameter("frame_id", "zed2_left_camera_optical_frame");
  node->declare_parameter("yolo_model_path", "./yolov8s-seg.onnx");

  std::string suffix = node->get_parameter("suffix").as_string();
  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  std::string point_cloud_topic_name =
      node->get_parameter("point_cloud_topic_name").as_string() + suffix;
  RCLCPP_INFO(node->get_logger(), "Point cloud topic: %s",
              point_cloud_topic_name.c_str());
  std::string full_point_cloud_topic_name =
      node->get_parameter("full_point_cloud_topic_name").as_string() + suffix;
  std::string image_topic_name =
      node->get_parameter("image_topic_name").as_string() + suffix;
  std::string resolution = node->get_parameter("resolution").as_string();
  if (resolution != "2K" && resolution != "FHD" && resolution != "HD") {
    std::cout << "Only '2K', 'FHD' and HD resolutions are supported"
              << std::endl;
    return -1;
  }
  std::string frame_id = node->get_parameter("frame_id").as_string() + suffix;
  RCLCPP_INFO(node->get_logger(), "Frame ID: %s", frame_id.c_str());
  RCLCPP_INFO(node->get_logger(), "Resolution: %s", resolution.c_str());
  bool camera_stream = node->get_parameter("camera_stream").as_bool();
  cv::dnn::Net net;
  Yolov8Seg yolov8Seg;
  if (!yolov8Seg.ReadModel(net, yolo_model_path, true)) {
    std::cout << "ReadModel failed, is the path correct?" << std::endl;
    return -1;
  }
  RCLCPP_INFO(node->get_logger(), "YOLOv8 model loaded");

  auto point_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      point_cloud_topic_name, 1);

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      full_point_cloud_pub;
  if (full_point_cloud_topic_name != "") {
    full_point_cloud_pub =
        node->create_publisher<sensor_msgs::msg::PointCloud2>(
            full_point_cloud_topic_name, 1);
    RCLCPP_INFO(node->get_logger(),
                "Full point cloud publisher created on topic %s",
                full_point_cloud_topic_name.c_str());
  }
  // create RGB camera publisher
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> image_pub;
  if (camera_stream) {
    RCLCPP_INFO(node->get_logger(), "RGB camera publisher created on topic %s",
                image_topic_name.c_str());
    image_pub =
        node->create_publisher<sensor_msgs::msg::Image>(image_topic_name, 1);
  }

  RCLCPP_INFO(node->get_logger(), "Point cloud publisher created on topic %s",
              point_cloud_topic_name.c_str());
  Camera zed;
  InitParameters init_parameters;
  if (resolution == "2K") {
    init_parameters.camera_resolution = RESOLUTION::HD2K;
  } else {
    if (resolution == "FHD") {
      init_parameters.camera_resolution = RESOLUTION::HD1080;
    } else {
      init_parameters.camera_resolution = RESOLUTION::HD720;
    }
  }
  init_parameters.camera_fps = 30;
  init_parameters.depth_mode = DEPTH_MODE::NEURAL;
  init_parameters.coordinate_system = COORDINATE_SYSTEM::IMAGE;
  // use meters
  init_parameters.coordinate_units = UNIT::METER;
  init_parameters.enable_image_validity_check = true;
  // init_parameters.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
  // init_parameters.svo_real_time_mode = true;

  parseArgsMonoCam(argc, argv, init_parameters);
  // disable self calibration
  // init_parameters.camera_disable_sel  f_calib = false;
  auto returned_state = zed.open(init_parameters);
  CalibrationParameters calibration_params =
      zed.getCameraInformation().camera_configuration.calibration_parameters;
  // Focal length of the left eye in pixels
  float focal_left_x = calibration_params.left_cam.fx;
  float focal_left_y = calibration_params.left_cam.fy;
  float cx = calibration_params.left_cam.cx;
  float cy = calibration_params.left_cam.cy;
  // First radial distortion coefficient
  double *dist = calibration_params.left_cam.disto;

  // reboot
  // zed.reboot(0);

  RCLCPP_INFO(node->get_logger(), "Focal length: %f %f", focal_left_x,
              focal_left_y);
  RCLCPP_INFO(node->get_logger(), "Principal point: %f %f", cx, cy);
  RCLCPP_INFO(node->get_logger(), "Distortion: %f %f %f %f", dist[0], dist[1],
              dist[2], dist[3]);
  // reset camera settings
  zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, VIDEO_SETTINGS_VALUE_AUTO);
  if (point_cloud_topic_name == "zed_point_cloud_2") {
    // increase exposure
    RCLCPP_INFO(node->get_logger(), "Increasing exposure");
    zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, 20);
  }
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

  unsigned int serial_num = 0;
  bool run = true;

  RuntimeParameters rt_params;
  rt_params.measure3D_reference_frame = REFERENCE_FRAME::CAMERA;

  cv::Mat image;
  auto point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  std::vector<OutputParams> output;

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

  // preallocate point cloud message
  auto pcMsg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  cv::namedWindow("YOLO Segmentation", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("RGB Image", cv::WINDOW_AUTOSIZE);
  sl::Resolution image_size =
      zed.getCameraInformation().camera_configuration.resolution;
  RCLCPP_INFO(node->get_logger(), "Serial number: %d",
              zed.getCameraInformation().serial_number);

  while (rclcpp::ok() && run) {
    auto err = zed.grab(rt_params);
    if (err != ERROR_CODE::SUCCESS) {
      RCLCPP_ERROR(node->get_logger(), "Grab failed: %s",
                   toString(err).c_str());
      continue;
    }
    sl::Mat sl_image;
    zed.retrieveImage(sl_image, VIEW::LEFT);
    sl::Mat point_cloud;
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZBGRA);

    cv::Mat cvImageTemp(sl_image.getHeight(), sl_image.getWidth(),
                        (sl_image.getChannels() == 4) ? CV_8UC4 : CV_8UC1,
                        sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::Mat cvImage = cvImageTemp.clone();
    if (cvImage.channels() == 4) {
      cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);
    }

    if (full_point_cloud_topic_name != "") {
      pcMsg->width = point_cloud.getWidth();
      pcMsg->height = point_cloud.getHeight();
      pcMsg->is_dense = false;
      pcMsg->is_bigendian = false;
      pcMsg->fields.clear();
      pcMsg->fields.push_back(x_field);
      pcMsg->fields.push_back(y_field);
      pcMsg->fields.push_back(z_field);
      pcMsg->fields.push_back(rgb_field);
      pcMsg->point_step = 16;
      pcMsg->row_step = pcMsg->point_step * pcMsg->width;
      pcMsg->data.resize(pcMsg->row_step * pcMsg->height);
      pcMsg->header.stamp = node->now();
      pcMsg->header.frame_id = frame_id;
      sensor_msgs::PointCloud2Modifier modifier2(*(pcMsg.get()));

      sl::Vector4<float> *cpu_cloud = point_cloud.getPtr<sl::float4>();
      float *ptCloudPtr = reinterpret_cast<float *>(&pcMsg->data[0]);
      int ptsCount = point_cloud.getWidth() * point_cloud.getHeight();

      std::memcpy(ptCloudPtr,
                  std::launder(reinterpret_cast<float *>(cpu_cloud)),
                  ptsCount * 4 * sizeof(float));

      // publish full point cloud
      full_point_cloud_pub->publish(*pcMsg);
    }

    if (resolution == "2K") {
      show_resized_fhd(cvImage, "RGB Image");
    } else {
      show_resized_fhd(cvImage, "RGB Image");
    }
    if (camera_stream) {
      publish_image(image_pub, cvImage, frame_id);
    }

#if DISPLAY_OGL
    viewer.updateData(bodies, cam_pose.pose_data);
#endif

    if (!yolov8Seg.Detect(cvImage, net, output)) {
      std::cout << "Detect failed" << std::endl;
      continue;
    }
    if (output.size() == 0) {
      std::cout << "No objects detected" << std::endl;
      continue;
    }
    int best_idx = -1;
    double max_mask_area = -1.0;
    cv::Mat biggest_component_mask, best_contour_visualization;

    // Create a new mask containing only the largest connected
    // component
    for (int it = 0; it < output.size(); it++) {
      if (output[it].id == 0) { // If it's a person
        cv::Mat mask = output[it].boxMask.clone();
        if (mask.empty())
          continue;

        // Ensure it's a binary mask
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);

        // Convert to 3-channel for visualization
        cv::Mat contour_visualization;
        cv::cvtColor(mask, contour_visualization, cv::COLOR_GRAY2BGR);

        // Find contours (connected components)
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        // Find the largest connected component
        double max_contour_area = 0.0;
        int largest_contour_idx = -1;
        for (size_t j = 0; j < contours.size(); j++) {
          double area = cv::contourArea(contours[j]);
          if (area > max_contour_area) {
            max_contour_area = area;
            largest_contour_idx = j;
          }
        }

        cv::Mat largest_component_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        if (largest_contour_idx != -1) {
          // Draw ONLY the largest contour in GREEN
          cv::drawContours(contour_visualization, contours, largest_contour_idx,
                           cv::Scalar(0, 255, 0), 2);

          // Fill the largest contour into the mask
          cv::drawContours(largest_component_mask, contours,
                           largest_contour_idx, cv::Scalar(255), cv::FILLED);
        }

        // Calculate area of the largest component
        double mask_area = cv::countNonZero(largest_component_mask);
        if (mask_area > max_mask_area) {
          max_mask_area = mask_area;
          best_idx = it;
          biggest_component_mask = largest_component_mask.clone();
          // best_contour_visualization =
          contour_visualization.clone();
        }
      }
    }

    if (best_idx != -1) {
      output[best_idx].boxMask = biggest_component_mask.clone();
    }

    if (best_idx == -1) {
      // No human detected
      // show_resized_img(cvImage, 0.7, "video");
      continue;
    }

    cv::Rect box = output[best_idx].box;
    cv::Mat boxMask = output[best_idx].boxMask;
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::Mat erodedMask;
    cv::erode(boxMask, erodedMask, kernel);
    boxMask = erodedMask; // Use eroded mask for
    // processing

    // continue;
    // std::string data_to_send = getJson(zed, bodies,
    // closest_body,
    // body_tracker_params.body_format)
    //                                .dump();
    // sock.sendTo(data_to_send.data(), data_to_send.size(),
    // servAddress,
    //             servPort);
    int bb_x_min = box.x;
    int bb_y_min = box.y;
    int bb_x_max = box.x + box.width;
    int bb_y_max = box.y + box.height;

    auto ros_pointcloud = sensor_msgs::msg::PointCloud2();
    ros_pointcloud.header.stamp = node->now();
    ros_pointcloud.header.frame_id = frame_id;

    ros_pointcloud.width = bb_x_max - bb_x_min;
    ros_pointcloud.height = bb_y_max - bb_y_min;
    ros_pointcloud.is_dense = false;
    ros_pointcloud.is_bigendian = false;

    ros_pointcloud.fields.push_back(x_field);
    ros_pointcloud.fields.push_back(y_field);
    ros_pointcloud.fields.push_back(z_field);
    ros_pointcloud.fields.push_back(rgb_field);
    ros_pointcloud.point_step = 16;
    ros_pointcloud.row_step = ros_pointcloud.point_step * ros_pointcloud.width;
    ros_pointcloud.data.resize(ros_pointcloud.row_step * ros_pointcloud.height);
    ros_pointcloud.is_dense = false;

    float *data = reinterpret_cast<float *>(ros_pointcloud.data.data());
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
          // if (erodedMask.at<uchar>(y - bb_y_min, x - bb_x_min)
          // == 0) {
          //   pixel[0] = 255;
          //   continue;
          // } else {
          point_cloud.getValue(x, y, &point3D);
          if (index < ros_pointcloud.width * ros_pointcloud.height) {
            data[index * 4 + 0] = point3D.x;
            data[index * 4 + 1] = point3D.y;
            data[index * 4 + 2] = point3D.z;
            uint32_t rgba = *reinterpret_cast<uint32_t *>(&point3D.w);
            std::memcpy(&data[index * 4 + 3], &rgba, 4);
            index++;
          }
          pixel[2] = 255;
          // }
        }
      }
    }
    //   // draw bounding box
    cv::rectangle(cvImage, cv::Point(bb_x_min, bb_y_min),
                  cv::Point(bb_x_max, bb_y_max), cv::Scalar(255, 0, 0), 3);
    show_resized_img(cvImage, 0.4, "YOLO Segmentation");
    point_cloud_pub->publish(ros_pointcloud);
  }
  // catch (SocketException &e) {
  //   std::cerr << e.what() << std::endl;
  // }
  // sl::sleep_ms(10);
  // }

  // #if DISPLAY_OGL
  //     run = viewer.isAvailable();
  // #endif

  // }

#if DISPLAY_OGL
  viewer.exit();
#endif

  // bodies.body_list.clear();

  // zed.disableBodyTracking();
  // zed.disablePositionalTracking();
  zed.close();

  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
