

#ifdef __GNUC__
// Avoid warnings
#pragma GCC system_header
#include <sl/Camera.hpp>
#include "json.hpp"
#endif
using namespace sl;
int clip(const int &n, const int &lower, const int &upper)
{
  return std::max(lower, std::min(n, upper));
}

int getOCVtype(sl::MAT_TYPE type)
{
  int cv_type = -1;
  switch (type)
  {
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
cv::Mat slMat2cvMat(sl::Mat &input)
{
  // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer
  // from sl::Mat (getPtr<T>()) cv::Mat and sl::Mat will share a single memory
  // structure
  return cv::Mat(
      input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
      input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}
/// ----------------------------------------------------------------------------
/// ----------------------------------------------------------------------------
/// ----------------------------- DATA FORMATTING
/// ------------------------------
/// ----------------------------------------------------------------------------
/// ----------------------------------------------------------------------------

// void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
//   std::cout << "[Sample]";
//   if (err_code != ERROR_CODE::SUCCESS) std::cout << "[Error]";
//   std::cout << " " << msg_prefix << " ";
//   if (err_code != ERROR_CODE::SUCCESS) {
//     std::cout << " | " << toString(err_code) << " : ";
//     std::cout << toVerbose(err_code);
//   }
//   if (!msg_suffix.empty()) std::cout << " " << msg_suffix;
//   std::cout << std::endl;
// }
// If the sender encounter NaN values, it sends 0 instead.
nlohmann::json bodyDataToJson(sl::BodyData body)
{
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
  for (auto &i : body.bounding_box)
  {
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
  for (auto &i : body.keypoint)
  {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 0 : i.x / 1000;
    e["y"] = isnan(i.y) ? 0 : i.y / 1000;
    e["z"] = isnan(i.z) ? 0 : i.z / 1000;
    res["keypoint"].push_back(e);
  }

  res["keypoint_confidence"] = nlohmann::json::array();
  for (auto &i : body.keypoint_confidence)
  {
    res["keypoint_confidence"].push_back(isnan(i) ? 0 : i);
  }
  res["local_position_per_joint"] = nlohmann::json::array();
  for (auto &i : body.local_position_per_joint)
  {
    nlohmann::json e;
    e["x"] = isnan(i.x) ? 0 : i.x / 1000;
    e["y"] = isnan(i.y) ? 0 : i.y / 1000;
    e["z"] = isnan(i.z) ? 0 : i.z / 1000;
    res["local_position_per_joint"].push_back(e);
  }
  res["local_orientation_per_joint"] = nlohmann::json::array();
  for (auto &i : body.local_orientation_per_joint)
  {
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
// Create the json sent to the clients
nlohmann::json getJson(sl::Camera &pcamera, sl::Bodies &bodies,
                       sl::BODY_FORMAT body_format)
{
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

  for (auto &body : bodies.body_list)
  {
    bodyData["body_list"].push_back(bodyDataToJson(body));
  }
  j["bodies"] = bodyData;

  return j;
}

// send one skeleton at a time
nlohmann::json getJson(sl::Camera &pcamera, sl::Bodies &bodies, int id,
                       sl::BODY_FORMAT body_format)
{
  Pose camp;
  pcamera.getPosition(camp);
  /// std::std::cout << "camera position : " << camp.getTranslation() << std::std::endl;

  nlohmann::json j;

  nlohmann::json bodyData;

  bodyData["body_format"] = body_format;
  bodyData["is_new"] = (int)bodies.is_new;
  bodyData["is_tracked"] = (int)bodies.is_tracked;
  bodyData["timestamp"] = bodies.timestamp.data_ns;

  bodyData["nb_object"] = bodies.body_list.size();
  bodyData["body_list"] = nlohmann::json::array();

  if (id < bodies.body_list.size())
  {
    bodyData["body_list"].push_back(bodyDataToJson(bodies.body_list[id]));
  }
  // std::cout << "Global orientation: " << bodyData["body_list"][0]["global_root_orientation"] << std::endl;
  // std::cout << "Global position: " << bodyData["body_list"][0]["position"] << std::endl;

  j["bodies"] = bodyData;

  return j;
}



// Parse command line arguments for mono-camera.
void parseArgsMonoCam(int argc, char **argv, InitParameters &param)
{
  if (argc > 1 && std::string(argv[1]).find(".svo") != std::string::npos)
  {
    // SVO input mode
    param.input.setFromSVOFile(argv[1]);
    std::cout << "[Sample] Using SVO File input: " << argv[1] << std::endl;
  }
  else if (argc > 1 && std::string(argv[1]).find(".svo") == std::string::npos)
  {
    std::string arg = std::string(argv[1]);
    unsigned int a, b, c, d, port;
    if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5)
    {
      // Stream input mode - IP + port
      std::string ip_adress = std::to_string(a) + "." + std::to_string(b) + "." +
                         std::to_string(c) + "." + std::to_string(d);
      param.input.setFromStream(String(ip_adress.c_str()), port);
      std::cout << "[Sample] Using Stream input, IP : " << ip_adress
           << ", port : " << port << std::endl;
    }
    else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4)
    {
      // Stream input mode - IP only
      param.input.setFromStream(String(argv[1]));
      std::cout << "[Sample] Using Stream input, IP : " << argv[1] << std::endl;
    }
    else if (arg.find("HD2K") != std::string::npos)
    {
      param.camera_resolution = RESOLUTION::HD2K;
      std::cout << "[Sample] Using Camera in resolution HD2K" << std::endl;
    }
    else if (arg.find("HD1080") != std::string::npos)
    {
      param.camera_resolution = RESOLUTION::HD1080;
      std::cout << "[Sample] Using Camera in resolution HD1080" << std::endl;
    }
    else if (arg.find("HD720") != std::string::npos)
    {
      param.camera_resolution = RESOLUTION::HD720;
      std::cout << "[Sample] Using Camera in resolution HD720" << std::endl;
    }
    else if (arg.find("VGA") != std::string::npos)
    {
      param.camera_resolution = RESOLUTION::VGA;
      std::cout << "[Sample] Using Camera in resolution VGA" << std::endl;
    }
  }
}