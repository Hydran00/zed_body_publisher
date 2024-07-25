// ZED include
#include <sl/Camera.hpp>
// OpenCV
#include <opencv2/opencv.hpp>
// ROS2 includes
#include <rclcpp/rclcpp.hpp>

#include "GLViewer.hpp"
#include "PracticalSocket.h"
#include "json.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"


using namespace sl;
using namespace std::chrono_literals;

nlohmann::json getJson(sl::Camera& pcamera, sl::Bodies& bodies,
                       sl::BODY_FORMAT body_format);
nlohmann::json getJson(sl::Camera& pcamera, sl::Bodies& bodies, int id,
                       sl::BODY_FORMAT body_format);

nlohmann::json bodyDataToJson(sl::BodyData body);
void parseArgsMonoCam(int argc, char** argv, InitParameters& param);

int clip(const int& n, const int& lower, const int& upper) {
  return std::max(lower, std::min(n, upper));
}

int getOCVtype(sl::MAT_TYPE type) {
  int cv_type = -1;
  switch (type) {
    case MAT_TYPE::F32_C1:
      cv_type = CV_32FC1;
      break;
    case MAT_TYPE::F32_C2:
      cv_type = CV_32FC2;
      break;
    case MAT_TYPE::F32_C3:
      cv_type = CV_32FC3;
      break;
    case MAT_TYPE::F32_C4:
      cv_type = CV_32FC4;
      break;
    case MAT_TYPE::U8_C1:
      cv_type = CV_8UC1;
      break;
    case MAT_TYPE::U8_C2:
      cv_type = CV_8UC2;
      break;
    case MAT_TYPE::U8_C3:
      cv_type = CV_8UC3;
      break;
    case MAT_TYPE::U8_C4:
      cv_type = CV_8UC4;
      break;
    default:
      break;
  }
  return cv_type;
}
cv::Mat slMat2cvMat(sl::Mat& input) {
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer
  // from sl::Mat (getPtr<T>()) cv::Mat and sl::Mat will share a single memory
  // structure
  return cv::Mat(
      input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
      input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

int main(int argc, char** argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("zed_publisher");
  auto point_cloud_pub =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 1);

  Camera zed;
  InitParameters init_parameters;
  init_parameters.camera_resolution = RESOLUTION::HD720;
  init_parameters.camera_fps = 30;
  init_parameters.depth_mode = DEPTH_MODE::NEURAL;
  init_parameters.coordinate_system = COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
  init_parameters.svo_real_time_mode = true;

  parseArgsMonoCam(argc, argv, init_parameters);

  auto returned_state = zed.open(init_parameters);
  if (returned_state != ERROR_CODE::SUCCESS) {
    zed.close();
    return EXIT_FAILURE;
  }

  PositionalTrackingParameters positional_tracking_parameters;
  positional_tracking_parameters.set_floor_as_origin = true;

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
  body_tracker_parameters_rt.minimum_keypoints_threshold = 7;

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
  while (rclcpp::ok() && run) {
    auto err = zed.grab(rt_params);
    if (err == ERROR_CODE::SUCCESS) {
      sl::Mat sl_image;
      zed.retrieveImage(sl_image, VIEW::LEFT);

      cv::Mat cvImage(sl_image.getHeight(), sl_image.getWidth(),
                      (sl_image.getChannels() == 1) ? CV_8UC1 : CV_8UC4,
                      sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));
      // draw bbox
      cv::Mat resized = cvImage.clone();
      cv::resize(resized, resized, cv::Size(), 0.5, 0.5);
      cv::imshow("video", resized);
      cv::waitKey(1);
      zed.retrieveBodies(bodies, body_tracker_parameters_rt);
      if(bodies.body_list.size() == 0) {
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

          auto mask = bodies.body_list[closest_body].mask;

          // if mask is empty, skip
          if (mask.getWidth() == 0 || mask.getHeight() == 0) {
            std::cout << "Mask is empty, skip" << std::endl;
            continue;
          }

          // Bounding box coordinate
          int bb_x_min = bodies.body_list[closest_body].bounding_box_2d[0][0];
          int bb_y_min = bodies.body_list[closest_body].bounding_box_2d[0][1];
          int bb_x_max = bodies.body_list[closest_body].bounding_box_2d[2][0];
          int bb_y_max = bodies.body_list[closest_body].bounding_box_2d[2][1];

          sl::Mat body_point_cloud(bb_x_max - bb_x_min, bb_y_max - bb_y_min,
                                   MAT_TYPE::F32_C4, MEM::CPU);

          auto ros_pointcloud = sensor_msgs::msg::PointCloud2();
          ros_pointcloud.header.stamp = node->now();
          ros_pointcloud.header.frame_id = "map";
          // use bbox to mask the point cloud

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

          float* data = reinterpret_cast<float*>(ros_pointcloud.data.data());
          sl::float4 point3D;


 
          sl::uchar1 mask_value;
          int index = 0;
          for (int y = bb_y_min; y < bb_y_max; y++) {
            for (int x = bb_x_min; x < bb_x_max; x++) {
              // get the mask value of the pixel
              if (mask.getValue(x - bb_x_min, y - bb_y_min, &mask_value) ==
                  ERROR_CODE::SUCCESS) {
                if (int(mask_value) != 255) {
                  continue;
                } else {
                  // mask original image pixel
                  cv::Vec4b& pixel = cvImage.at<cv::Vec4b>(y, x);
                  pixel[0] = 0;
                  pixel[1] = 0;
                  pixel[2] = 255;
                  point_cloud.getValue(x, y, &point3D);
                  if (index < ros_pointcloud.width * ros_pointcloud.height) {
                    data[index * 4 + 0] = point3D.x / 1000.0;
                    data[index * 4 + 1] = point3D.y / 1000.0;
                    data[index * 4 + 2] = point3D.z / 1000.0;
                    uint32_t rgb = *reinterpret_cast<uint32_t*>(&point3D.w);
                    // convert from ABGR to RGBA
                    rgb = ((rgb & 0x000000FF) << 16) |
                          ((rgb & 0x0000FF00)) | ((rgb & 0x00FF0000) >> 16);
                    std::memcpy(&data[index * 4 + 3], &rgb, 4);
                    index++;
                  }
                }
              }
            }
          }
          cv::resize(cvImage, cvImage, cv::Size(), 0.5, 0.5);
          cv::imshow("video", cvImage);
          cv::waitKey(1);
          point_cloud_pub->publish(ros_pointcloud);
        } catch (SocketException& e) {
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
/// ----------------------------------------------------------------------------
/// ----------------------------------------------------------------------------
/// ----------------------------- DATA FORMATTING
/// ------------------------------
/// ----------------------------------------------------------------------------
/// ----------------------------------------------------------------------------

// void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
//   cout << "[Sample]";
//   if (err_code != ERROR_CODE::SUCCESS) cout << "[Error]";
//   cout << " " << msg_prefix << " ";
//   if (err_code != ERROR_CODE::SUCCESS) {
//     cout << " | " << toString(err_code) << " : ";
//     cout << toVerbose(err_code);
//   }
//   if (!msg_suffix.empty()) cout << " " << msg_suffix;
//   cout << endl;
// }

// Create the json sent to the clients
nlohmann::json getJson(sl::Camera& pcamera, sl::Bodies& bodies,
                       sl::BODY_FORMAT body_format) {
  Pose camp;
  pcamera.getPosition(camp);
  nlohmann::json j;

  nlohmann::json bodyData;

  bodyData["body_format"] = body_format;
  bodyData["is_new"] = (int)bodies.is_new;
  bodyData["is_tracked"] = (int)bodies.is_tracked;
  bodyData["timestamp"] = bodies.timestamp.data_ns;

  bodyData["nb_object"] = bodies.body_list.size();
  bodyData["body_list"] = nlohmann::json::array();

  for (auto& body : bodies.body_list) {
    bodyData["body_list"].push_back(bodyDataToJson(body));
  }

  j["bodies"] = bodyData;

  return j;
}

// send one skeleton at a time
nlohmann::json getJson(sl::Camera& pcamera, sl::Bodies& bodies, int id,
                       sl::BODY_FORMAT body_format) {
  Pose camp;
  pcamera.getPosition(camp);
  /// std::cout << "camera position : " << camp.getTranslation() << std::endl;

  nlohmann::json j;

  nlohmann::json bodyData;

  bodyData["body_format"] = body_format;
  bodyData["is_new"] = (int)bodies.is_new;
  bodyData["is_tracked"] = (int)bodies.is_tracked;
  bodyData["timestamp"] = bodies.timestamp.data_ns;

  bodyData["nb_object"] = bodies.body_list.size();
  bodyData["body_list"] = nlohmann::json::array();

  if (id < bodies.body_list.size()) {
    bodyData["body_list"].push_back(bodyDataToJson(bodies.body_list[id]));
  }

  j["bodies"] = bodyData;

  return j;
}

// If the sender encounter NaN values, it sends 0 instead.
nlohmann::json bodyDataToJson(sl::BodyData body) {
  nlohmann::json res;

  res["id"] = body.id;
  // res["unique_object_id"] = body.unique_object_id.get();
  res["tracking_state"] = body.tracking_state;
  res["action_state"] = body.action_state;
  res["position"] = nlohmann::json::object();
  res["position"]["x"] = isnan(body.position.x) ? 0 : body.position.x / 1000;
  res["position"]["y"] = isnan(body.position.y) ? 0 : body.position.y / 1000;
  res["position"]["z"] = isnan(body.position.z) ? 0 : body.position.z / 1000;

  res["velocity"] = nlohmann::json::object();
  res["velocity"]["x"] = isnan(body.velocity.x) ? 0 : body.velocity.x / 1000;
  res["velocity"]["y"] = isnan(body.velocity.y) ? 0 : body.velocity.y / 1000;
  res["velocity"]["z"] = isnan(body.velocity.z) ? 0 : body.velocity.z / 1000;

  res["confidence"] = isnan(body.confidence) ? 0 : body.confidence;
  res["bounding_box"] = nlohmann::json::array();
  for (auto& i : body.bounding_box) {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 0 : i.x / 1000;
    e["y"] = isnan(i.y) ? 0 : i.y / 1000;
    e["z"] = isnan(i.z) ? 0 : i.z / 1000;
    res["bounding_box"].push_back(e);
  }
  res["dimensions"] = nlohmann::json::object();
  res["dimensions"]["x"] =
      isnan(body.dimensions.x) ? 0 : body.dimensions.x / 1000;
  res["dimensions"]["y"] =
      isnan(body.dimensions.y) ? 0 : body.dimensions.y / 1000;
  res["dimensions"]["z"] =
      isnan(body.dimensions.z) ? 0 : body.dimensions.z / 1000;

  res["keypoint"] = nlohmann::json::array();
  for (auto& i : body.keypoint) {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 0 : i.x / 1000;
    e["y"] = isnan(i.y) ? 0 : i.y / 1000;
    e["z"] = isnan(i.z) ? 0 : i.z / 1000;
    res["keypoint"].push_back(e);
  }

  res["keypoint_confidence"] = nlohmann::json::array();
  for (auto& i : body.keypoint_confidence) {
    res["keypoint_confidence"].push_back(isnan(i) ? 0 : i);
  }
  res["local_position_per_joint"] = nlohmann::json::array();
  for (auto& i : body.local_position_per_joint) {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 0 : i.x / 1000;
    e["y"] = isnan(i.y) ? 0 : i.y / 1000;
    e["z"] = isnan(i.z) ? 0 : i.z / 1000;
    res["local_position_per_joint"].push_back(e);
  }
  res["local_orientation_per_joint"] = nlohmann::json::array();
  for (auto& i : body.local_orientation_per_joint) {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 42 : i.x;
    e["y"] = isnan(i.y) ? 42 : i.y;
    e["z"] = isnan(i.z) ? 42 : i.z;
    e["w"] = isnan(i.w) ? 42 : i.w;
    res["local_orientation_per_joint"].push_back(e);
  }
  res["global_root_orientation"] = nlohmann::json::object();
  res["global_root_orientation"]["x"] = isnan(body.global_root_orientation.x)
                                            ? 0
                                            : body.global_root_orientation.x;
  res["global_root_orientation"]["y"] = isnan(body.global_root_orientation.y)
                                            ? 0
                                            : body.global_root_orientation.y;
  res["global_root_orientation"]["z"] = isnan(body.global_root_orientation.z)
                                            ? 0
                                            : body.global_root_orientation.z;
  res["global_root_orientation"]["w"] = isnan(body.global_root_orientation.w)
                                            ? 0
                                            : body.global_root_orientation.w;
  return res;
}

// Parse command line arguments for mono-camera.
void parseArgsMonoCam(int argc, char** argv, InitParameters& param) {
  if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
    // SVO input mode
    param.input.setFromSVOFile(argv[1]);
    cout << "[Sample] Using SVO File input: " << argv[1] << endl;
  } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
    string arg = string(argv[1]);
    unsigned int a, b, c, d, port;
    if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
      // Stream input mode - IP + port
      string ip_adress = to_string(a) + "." + to_string(b) + "." +
                         to_string(c) + "." + to_string(d);
      param.input.setFromStream(String(ip_adress.c_str()), port);
      cout << "[Sample] Using Stream input, IP : " << ip_adress
           << ", port : " << port << endl;
    } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
      // Stream input mode - IP only
      param.input.setFromStream(String(argv[1]));
      cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
    } else if (arg.find("HD2K") != string::npos) {
      param.camera_resolution = RESOLUTION::HD2K;
      cout << "[Sample] Using Camera in resolution HD2K" << endl;
    } else if (arg.find("HD1080") != string::npos) {
      param.camera_resolution = RESOLUTION::HD1080;
      cout << "[Sample] Using Camera in resolution HD1080" << endl;
    } else if (arg.find("HD720") != string::npos) {
      param.camera_resolution = RESOLUTION::HD720;
      cout << "[Sample] Using Camera in resolution HD720" << endl;
    } else if (arg.find("VGA") != string::npos) {
      param.camera_resolution = RESOLUTION::VGA;
      cout << "[Sample] Using Camera in resolution VGA" << endl;
    }
  }
}