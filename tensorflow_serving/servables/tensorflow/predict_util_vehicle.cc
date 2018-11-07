/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/tensorflow/predict_util_vehicle.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "absl/strings/str_join.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow_serving/core/servable_handle.h"

#define max(a, b) ((a) > (b) ? (a):(b))
#define min(a, b) ((a) < (b) ? (a):(b))
#define VEHICLE_DETECT_THRESH 0.5
#define PLATE_DETECT_THRESH 0.5
#define PLATE_POINT_THRESH 0.9
#define COLOR_TYPE_HEIGHT 224
#define COLOR_TYPE_WIDTH 224
#define BRAND_HEIGHT 224
#define BRAND_WIDTH 224
#define PLATE_DETECT_HEIGHT 320
#define PLATE_DETECT_WIDTH 320
#define PLATE_ENLARGE_HEIGHT 0.3
#define PLATE_ENLARGE_WIDTH 0.1
#define PLATE_MIN_HEIGHT 10
#define PLATE_MIN_WIDTH 30
#define PLATE_POINT_HEIGHT 32
#define PLATE_POINT_WIDTH 96
#define PLATE_REC_HEIGHT 40
#define PLATE_REC_WIDTH 250
#define PLATE_CHAR_NUM 74
#define PLATE_PROVINCE_NUM 31
#define PLATE_MAX_LENGTH 8 


namespace tensorflow {
namespace serving {
namespace {

// Returns the keys in the map as a comma delimited string. Useful for debugging
// or when returning error messages.
// e.g. returns "key1, key2, key3".
string MapKeysToString(const google::protobuf::Map<string, tensorflow::TensorInfo>& map) 
{
  string result = "";
  for (const auto& i : map) {
    if (result.empty()) {
      result += i.first;
    } else {
      result += ", " + i.first;
    }
  }
  return result;
}

Status VerifySignature(const SignatureDef& signature) 
{
  if (signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    return errors::Internal(strings::StrCat(
        "Expected prediction signature method_name to be one of {",
        kPredictMethodName, ", ", kClassifyMethodName, ", ", kRegressMethodName,
        "}. Was: ", signature.method_name()));
  }
  return Status::OK();
}

template <typename T>
std::vector<string> get_map_keys(const T& proto_map) 
{
  std::vector<string> keys;
  for (auto it : proto_map) {
    keys.push_back(it.first);
  }
  return keys;
}

Status VerifyRequestInputsSize(const SignatureDef& signature,
                               const PredictRequest& request) 
{
  if (request.inputs().size() != signature.inputs().size()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        absl::StrCat(
            "input size does not match signature: ", request.inputs().size(),
            "!=", signature.inputs().size(), " len([",
            absl::StrJoin(get_map_keys(request.inputs()), ","), "]) != len([",
            absl::StrJoin(get_map_keys(signature.inputs()), ","), "])"));
  }
  return Status::OK();
}

// Validate a SignatureDef to make sure it's compatible with prediction, and
// if so, populate the input and output tensor names.
Status PreProcessPrediction(const SignatureDef& signature,
                            const PredictRequest& request,
                            std::vector<std::pair<string, Tensor>>* inputs,
                            std::vector<string>* output_tensor_names,
                            std::vector<string>* output_tensor_aliases) 
{
  TF_RETURN_IF_ERROR(VerifySignature(signature));
  TF_RETURN_IF_ERROR(VerifyRequestInputsSize(signature, request));
  for (auto& input : request.inputs()) {
    const string& alias = input.first;
    auto iter = signature.inputs().find(alias);
    if (iter == signature.inputs().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          strings::StrCat("input tensor alias not found in signature: ", alias,
                          ". Inputs expected to be in the set {",
                          MapKeysToString(signature.inputs()), "}."));
    }
    Tensor tensor;
    if (!tensor.FromProto(input.second)) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "tensor parsing error: " + alias);
    }
    inputs->emplace_back(std::make_pair(iter->second.name(), tensor));
  }

  // Prepare run target.
  std::set<string> seen_outputs;
  std::vector<string> output_filter(request.output_filter().begin(),
                                    request.output_filter().end());
  for (auto& alias : output_filter) {
    auto iter = signature.outputs().find(alias);
    if (iter == signature.outputs().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          strings::StrCat("output tensor alias not found in signature: ", alias,
                          " Outputs expected to be in the set {",
                          MapKeysToString(signature.outputs()), "}."));
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "duplicate output tensor alias: " + alias);
    }
    seen_outputs.insert(alias);
    output_tensor_names->emplace_back(iter->second.name());
    output_tensor_aliases->emplace_back(alias);
  }
  // When no output is specified, fetch all output tensors specified in
  // the signature.
  if (output_tensor_names->empty()) {
    for (auto& iter : signature.outputs()) {
      output_tensor_names->emplace_back(iter.second.name());
      output_tensor_aliases->emplace_back(iter.first);
    }
  }
  return Status::OK();
}

// Validate results and populate a PredictResponse.
Status PostProcessPredictionResult(
    const SignatureDef& signature,
    const std::vector<string>& output_tensor_aliases,
    const std::vector<Tensor>& output_tensors, PredictResponse* response) 
{
  // Validate and return output.
  if (output_tensors.size() != output_tensor_aliases.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }
  for (int i = 0; i < output_tensors.size(); i++) {
    output_tensors[i].AsProtoField(
        &((*response->mutable_outputs())[output_tensor_aliases[i]]));
  }
  return Status::OK();
}

}  // namespace


struct CW_VEHICLE{
  int batch_id;
  
  float car_coord[4]; //left_top_x, left_top_y, right_bottom_x, right_bottom_y
  float car_score;  
  int color_id;
  float color_score;  
  int type_id;
  float type_score;
  int brand_id;
  float brand_score;

  bool has_plate;
  float platedet_coord[4]; //left_top_x, left_top_y, right_bottom_x, right_bottom_y
  float platedet_score;
  float plateenlarge_coord[4]; //platepoint input
  cv::Rect enlarge_plate_rect;
  float platepoint_coord[8]; //clockwise x,y
  float platepoint_score[4]; //clockwise
  int platerec_id[PLATE_MAX_LENGTH];
  float platerec_score;
};


std::vector<cv::Mat> tensorproto2imagevec(const TensorProto& tensor_proto)
{
  Tensor origin_tensor;
  origin_tensor.FromProto(tensor_proto);
  unsigned char *data_pointer = origin_tensor.flat<unsigned char>().data();
  unsigned char *moving_data_pointer = data_pointer;

  int batch_size = origin_tensor.dim_size(0);
  int height = origin_tensor.dim_size(1);
  int width = origin_tensor.dim_size(2);
  int channels = origin_tensor.dim_size(3);

  std::vector<cv::Mat> image_vec;
  for(int batch_id = 0; batch_id < batch_size; batch_id++){
    int target_image_shape[2] = {height, width};
    cv::Mat target_image(2, target_image_shape, CV_8UC3, moving_data_pointer);
    image_vec.push_back(target_image.clone());
    moving_data_pointer += height * width * channels;
  }

  return image_vec;

}


TensorProto imagevec2tensorproto(const std::vector<cv::Mat>& image_vec)
{
  int batch_size = image_vec.size();
  int net_height = image_vec[0].rows;
  int net_width = image_vec[0].cols;
  int net_channels = image_vec[0].channels();

  // get images data
  int buffer_size = sizeof(unsigned char) * batch_size * net_height * net_width * net_channels;
  unsigned char *data_pointer = (unsigned char *)malloc(buffer_size);
  unsigned char *mv_data_pointer = data_pointer;
  for(int batch_id = 0; batch_id < batch_size; batch_id++){
    cv::Mat origin_image = image_vec[batch_id];
    unsigned char *origin_image_data = origin_image.data;
    int image_buffer_size = sizeof(unsigned char) * origin_image.rows * origin_image.cols * origin_image.channels();       
    memcpy(mv_data_pointer, origin_image_data, image_buffer_size);
    mv_data_pointer += image_buffer_size;
  }

  // get tensor
  TensorProto tensor_proto;
  Tensor target_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape({batch_size, net_height, net_width, net_channels}));
  memcpy(target_tensor.flat<unsigned char>().data(), data_pointer, buffer_size);
  target_tensor.AsProtoField(&tensor_proto);

  mv_data_pointer = NULL;
  std::free(data_pointer);
  data_pointer = NULL;

  return tensor_proto;
}


Status PostProcessVehicleDetection(const std::vector<cv::Mat>& origin_image_vec,
                                 std::vector<CW_VEHICLE>& vehicle_result_vec,
                                 const std::vector<string>& output_tensor_aliases,
                                 const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  Tensor output_boxes;
  Tensor output_scores;
  Tensor output_classes;
  Tensor output_count;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_boxes") == 0){
      output_boxes = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_scores") == 0){
      output_scores = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_classes") == 0){
      output_classes = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_count") == 0){
      output_count = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, common detect output not find!");
    }
  }

  int batch_size = output_boxes.dim_size(0);
  int image_height = origin_image_vec[0].rows;
  int image_width = origin_image_vec[0].cols;

  auto output_boxes_mapped = output_boxes.tensor<float, 3>();
  auto output_scores_mapped = output_scores.tensor<float, 2>();            
  auto output_classes_mapped = output_classes.tensor<float, 2>();          
  auto output_count_mapped = output_count.tensor<float, 1>(); 

  for(int batch_id = 0; batch_id < batch_size; batch_id++){
    for(int object_id = 0; object_id < output_count_mapped(batch_id); object_id++){
      // object detect is car with confidence > VEHICLE_DETECT_THRESH
      if(output_classes_mapped(batch_id, object_id) == 3 && output_scores_mapped(batch_id, object_id) >= VEHICLE_DETECT_THRESH){
        float left_top_y = output_boxes_mapped(batch_id, object_id, 0);
        float left_top_x = output_boxes_mapped(batch_id, object_id, 1);
        float right_bottom_y = output_boxes_mapped(batch_id, object_id, 2);
        float right_bottom_x = output_boxes_mapped(batch_id, object_id, 3);

        // VLOG(0) << left_top_x << ", " << left_top_y << ", " << right_bottom_x << ", " << right_bottom_y;
        // VLOG(0) << image_height << ", " << image_width;

        CW_VEHICLE cw_vehicle;
        memset(&cw_vehicle, 0, sizeof(cw_vehicle));
        cw_vehicle.car_coord[1] = floor(left_top_y * image_height);
        cw_vehicle.car_coord[0] = floor(left_top_x * image_width);
        cw_vehicle.car_coord[3] = floor(right_bottom_y * image_height);
        cw_vehicle.car_coord[2] = floor(right_bottom_x * image_width);
        cw_vehicle.car_score = output_scores_mapped(batch_id, object_id);
        cw_vehicle.batch_id = batch_id;
        vehicle_result_vec.push_back(cw_vehicle);

        // VLOG(0) << "car_coord: " << cw_vehicle.car_coord[0] << ", " << cw_vehicle.car_coord[1] << ", " << cw_vehicle.car_coord[2] << ", " << cw_vehicle.car_coord[3];
        // VLOG(0) << "car_score: " << cw_vehicle.car_score;
      }   

    }
  }

  return Status::OK();

}


TensorProto GetVehicleColorTypeInput(const std::vector<cv::Mat>& origin_image_vec,
                                     std::vector<CW_VEHICLE>& vehicle_result_vec)
{
  // get image_vec
  int net_shape[2] = {COLOR_TYPE_HEIGHT, COLOR_TYPE_WIDTH};
  std::vector<cv::Mat> image_vec;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    CW_VEHICLE cw_vehicle = vehicle_result_vec[object_id];
    cv::Mat origin_image = origin_image_vec[cw_vehicle.batch_id].clone();

    // get roi
    cv::Rect roi_rect;
    roi_rect.x = cw_vehicle.car_coord[0];
    roi_rect.y = cw_vehicle.car_coord[1];
    roi_rect.width = cw_vehicle.car_coord[2] - cw_vehicle.car_coord[0];
    roi_rect.height = cw_vehicle.car_coord[3] - cw_vehicle.car_coord[1];
    cv::Mat roi = origin_image(roi_rect);

    // resize    
    cv::Mat resize_roi;
    float netw_div_imgw = (float)net_shape[1] / roi_rect.width;
    float neth_div_imgh = (float)net_shape[0] / roi_rect.height;
    cv::Size new_size(0,0);
    if(netw_div_imgw < neth_div_imgh){
      new_size.width = net_shape[1];
      new_size.height = floor(roi_rect.height * netw_div_imgw);
    }
    else{
      new_size.height = net_shape[0];
      new_size.width = floor(roi_rect.width * neth_div_imgh);      
    }
    cv::resize(roi, resize_roi, new_size);
    cv::Mat target_image = cv::Mat(net_shape[0], net_shape[1], CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect target_rect;
    target_rect.x = floor((net_shape[1] - new_size.width)/2);
    target_rect.y = floor((net_shape[0] - new_size.height)/2);
    target_rect.width = new_size.width;
    target_rect.height = new_size.height;
    resize_roi.copyTo(target_image(target_rect));

    image_vec.push_back(target_image);        
  }

  // get input_tensor_proto
  TensorProto input_tensor_proto = imagevec2tensorproto(image_vec);

  return input_tensor_proto;

}


Status PostProcessVehicleColorType(const std::vector<cv::Mat>& origin_image_vec,
                                 std::vector<CW_VEHICLE>& vehicle_result_vec,
                                 const std::vector<string>& output_tensor_aliases,
                                 const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  Tensor output_color_indices;
  Tensor output_color_confidences;
  Tensor output_type_indices;
  Tensor output_type_confidences;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_color_indices") == 0){
      output_color_indices = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_color_confidences") == 0){
      output_color_confidences = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_type_indices") == 0){
      output_type_indices = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_type_confidences") == 0){
      output_type_confidences = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, common detect output not find!");
    }
  }

  int batch_size = output_color_indices.dim_size(0);

  // top 5 result
  auto output_color_indices_mapped = output_color_indices.tensor<int, 2>();
  auto output_color_confidences_mapped = output_color_confidences.tensor<float, 2>();
  auto output_type_indices_mapped = output_type_indices.tensor<int, 2>();
  auto output_type_confidences_mapped = output_type_confidences.tensor<float, 2>();    

  for(int batch_id = 0; batch_id < batch_size; batch_id++){
    vehicle_result_vec[batch_id].color_id = output_color_indices_mapped(batch_id, 0);
    vehicle_result_vec[batch_id].color_score = output_color_confidences_mapped(batch_id, 0);
    vehicle_result_vec[batch_id].type_id = output_type_indices_mapped(batch_id, 0);
    vehicle_result_vec[batch_id].type_score = output_type_confidences_mapped(batch_id, 0);

    // VLOG(0) << "color_id: " << vehicle_result_vec[batch_id].color_id << " color_score: " << vehicle_result_vec[batch_id].color_score;
    // VLOG(0) << "type_id: " << vehicle_result_vec[batch_id].type_id << " type_score: " << vehicle_result_vec[batch_id].type_score;
  }

  return Status::OK();

}


TensorProto GetVehicleBrandInput(const std::vector<cv::Mat>& origin_image_vec,
                                 std::vector<CW_VEHICLE>& vehicle_result_vec)
{
  // get image_vec
  std::vector<cv::Mat> image_vec;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    CW_VEHICLE cw_vehicle = vehicle_result_vec[object_id];
    cv::Mat origin_image = origin_image_vec[cw_vehicle.batch_id].clone();

    // get roi
    cv::Rect roi_rect;
    roi_rect.x = cw_vehicle.car_coord[0];
    roi_rect.y = cw_vehicle.car_coord[1];
    roi_rect.width = cw_vehicle.car_coord[2] - cw_vehicle.car_coord[0];
    roi_rect.height = cw_vehicle.car_coord[3] - cw_vehicle.car_coord[1];
    cv::Mat roi = origin_image(roi_rect);

    // resize    
    cv::Mat resize_roi;
    cv::resize(roi, resize_roi, cv::Size(BRAND_WIDTH, BRAND_HEIGHT));

    image_vec.push_back(resize_roi);        
  }

  // get input_tensor_proto
  TensorProto input_tensor_proto = imagevec2tensorproto(image_vec);

  return input_tensor_proto;

}


Status PostProcessVehicleBrand(const std::vector<cv::Mat>& origin_image_vec,
                               std::vector<CW_VEHICLE>& vehicle_result_vec,
                               const std::vector<string>& output_tensor_aliases,
                               const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  // output of car brand
  Tensor output_indices;
  Tensor output_confidences;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_indices") == 0){
      output_indices = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_confidences") == 0){
      output_confidences = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, common detect output not find!");
    }
  }

  int batch_size = output_indices.dim_size(0);

  // top 5 result
  auto output_indices_mapped = output_indices.tensor<int, 2>();
  auto output_confidences_mapped = output_confidences.tensor<float, 2>();  

  for(int batch_id = 0; batch_id < batch_size; batch_id++){
    vehicle_result_vec[batch_id].brand_id = output_indices_mapped(batch_id, 0);
    vehicle_result_vec[batch_id].brand_score = output_confidences_mapped(batch_id, 0);

    // VLOG(0) << "brand_id: " << vehicle_result_vec[batch_id].brand_id << " brand_score: " << vehicle_result_vec[batch_id].brand_score;

  }

  return Status::OK();

}


TensorProto GetPlateDetectInput(const std::vector<cv::Mat>& origin_image_vec,
                                std::vector<CW_VEHICLE>& vehicle_result_vec)
{
  // get image_vec
  std::vector<cv::Mat> image_vec;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];
    cv::Mat origin_image = origin_image_vec[cw_vehicle.batch_id].clone();

    // get roi
    cv::Rect roi_rect;
    roi_rect.x = cw_vehicle.car_coord[0];
    roi_rect.y = cw_vehicle.car_coord[1];
    roi_rect.width = cw_vehicle.car_coord[2] - cw_vehicle.car_coord[0];
    roi_rect.height = cw_vehicle.car_coord[3] - cw_vehicle.car_coord[1];
    cv::Mat roi = origin_image(roi_rect);

    // resize    
    cv::Mat resize_roi;
    cv::resize(roi, resize_roi, cv::Size(PLATE_DETECT_WIDTH, PLATE_DETECT_HEIGHT));

    image_vec.push_back(resize_roi);        
  }

  // get input_tensor_proto
  TensorProto input_tensor_proto = imagevec2tensorproto(image_vec);

  return input_tensor_proto;

}


Status PostProcessPlateDetection(const std::vector<cv::Mat>& origin_image_vec,
                                 std::vector<CW_VEHICLE>& vehicle_result_vec,
                                 const std::vector<string>& output_tensor_aliases,
                                 const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  Tensor output_boxes;
  Tensor output_scores;
  Tensor output_classes;
  Tensor output_count;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_boxes") == 0){
      output_boxes = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_scores") == 0){
      output_scores = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_classes") == 0){
      output_classes = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_count") == 0){
      output_count = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, common detect output not find!");
    }
  }

  auto output_boxes_mapped = output_boxes.tensor<float, 3>();
  auto output_scores_mapped = output_scores.tensor<float, 2>();            
  auto output_classes_mapped = output_classes.tensor<int, 2>();          
  auto output_count_mapped = output_count.tensor<int, 0>(); 

  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    // object detect is the first one with confidence > PLATE_DETECT_THRESH
    if(output_scores_mapped(object_id, 0) >= PLATE_DETECT_THRESH){
      vehicle_result_vec[object_id].has_plate = true;
      CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];

      float car_left_top_x = cw_vehicle.car_coord[0];
      float car_left_top_y = cw_vehicle.car_coord[1];
      float car_height = cw_vehicle.car_coord[3] - cw_vehicle.car_coord[1];
      float car_width = cw_vehicle.car_coord[2] - cw_vehicle.car_coord[0];

      float left_top_y = output_boxes_mapped(object_id, 0, 0);
      float left_top_x = output_boxes_mapped(object_id, 0, 1);
      float right_bottom_y = output_boxes_mapped(object_id, 0, 2);
      float right_bottom_x = output_boxes_mapped(object_id, 0, 3);

      // VLOG(0) << "platedet_origin_coord: " << left_top_x << ", " << left_top_y << ", " << right_bottom_x << ", " << right_bottom_y;

      cw_vehicle.platedet_coord[1] = floor(left_top_y * car_height) + car_left_top_y;
      cw_vehicle.platedet_coord[0] = floor(left_top_x * car_width) + car_left_top_x;
      cw_vehicle.platedet_coord[3] = floor(right_bottom_y * car_height) + car_left_top_y;
      cw_vehicle.platedet_coord[2] = floor(right_bottom_x * car_width) + car_left_top_x;
      cw_vehicle.platedet_score = output_scores_mapped(object_id, 0);

      // VLOG(0) << "platedet_coord: " << cw_vehicle.platedet_coord[0] << ", " << cw_vehicle.platedet_coord[1] << ", " << cw_vehicle.platedet_coord[2] << ", " << cw_vehicle.platedet_coord[3];
      // VLOG(0) << "platedet_score: " << cw_vehicle.platedet_score;
    }
    else{
      vehicle_result_vec[object_id].has_plate = false;
    }   
  }

  return Status::OK();

}



TensorProto GetPlatePointInput(const std::vector<cv::Mat>& origin_image_vec,
                               std::vector<CW_VEHICLE>& vehicle_result_vec)
{
  // get image_vec
  std::vector<cv::Mat> image_vec;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    if(vehicle_result_vec[object_id].has_plate){
      CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];
      cv::Mat origin_image = origin_image_vec[cw_vehicle.batch_id].clone();

      //enlarge
      float plate_height = cw_vehicle.platedet_coord[3] - cw_vehicle.platedet_coord[1];
      float plate_width = cw_vehicle.platedet_coord[2] - cw_vehicle.platedet_coord[0];
      float margin_height = floor(plate_height * PLATE_ENLARGE_HEIGHT);
      float margin_width = floor(plate_width * PLATE_ENLARGE_WIDTH);
      cw_vehicle.plateenlarge_coord[0] = max(0, cw_vehicle.platedet_coord[0] - margin_width);
      cw_vehicle.plateenlarge_coord[1] = max(0, cw_vehicle.platedet_coord[1] - margin_height);
      cw_vehicle.plateenlarge_coord[2] = min(origin_image.cols, cw_vehicle.platedet_coord[2] + margin_width);
      cw_vehicle.plateenlarge_coord[3] = min(origin_image.rows, cw_vehicle.platedet_coord[3] + margin_height);
      cw_vehicle.enlarge_plate_rect.x = cw_vehicle.plateenlarge_coord[0];
      cw_vehicle.enlarge_plate_rect.y = cw_vehicle.plateenlarge_coord[1];
      cw_vehicle.enlarge_plate_rect.width = cw_vehicle.plateenlarge_coord[2] - cw_vehicle.plateenlarge_coord[0];
      cw_vehicle.enlarge_plate_rect.height = cw_vehicle.plateenlarge_coord[3] - cw_vehicle.plateenlarge_coord[1];    
      
      // VLOG(0) << "plateenlarge_coord: " << cw_vehicle.plateenlarge_coord[0] << ", " << cw_vehicle.plateenlarge_coord[1] << ", " << cw_vehicle.plateenlarge_coord[2] << ", " << cw_vehicle.plateenlarge_coord[3];

      //filter by size
      if((cw_vehicle.plateenlarge_coord[3] - cw_vehicle.plateenlarge_coord[1]) < PLATE_MIN_HEIGHT || (cw_vehicle.plateenlarge_coord[2] - cw_vehicle.plateenlarge_coord[0]) < PLATE_MIN_WIDTH){
        cw_vehicle.has_plate = false;
      }
      else{
        // get roi
        cv::Mat roi = origin_image(cw_vehicle.enlarge_plate_rect);

        // resize    
        cv::Mat resize_roi;
        cv::resize(roi, resize_roi, cv::Size(PLATE_POINT_WIDTH, PLATE_POINT_HEIGHT));

        image_vec.push_back(resize_roi);

      }
    }          
  }

  // get input_tensor_proto
  TensorProto input_tensor_proto = imagevec2tensorproto(image_vec);

  return input_tensor_proto;
}


Status PostProcessPlatePoint(const std::vector<cv::Mat>& origin_image_vec,
                             std::vector<CW_VEHICLE>& vehicle_result_vec,
                             const std::vector<string>& output_tensor_aliases,
                             const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  // output of plate point
  Tensor output_coordinates;
  Tensor output_confidences;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_coordinates") == 0){
      output_coordinates = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_confidences") == 0){
      output_confidences = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, common detect output not find!");
    }
  }

  auto output_coordinates_mapped = output_coordinates.tensor<float, 2>();
  auto output_confidences_mapped = output_confidences.tensor<float, 2>();

  int valid_count = 0;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){  
    if(vehicle_result_vec[object_id].has_plate){
      CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];

      // VLOG(0) << "platepoint_origin_coord: ";
      // for(int i = 0; i < 8; i++){
      //   VLOG(0) << output_coordinates_mapped(valid_count, i);
      // }

      float plate_shape[2] = {cw_vehicle.enlarge_plate_rect.width, cw_vehicle.enlarge_plate_rect.height};
      for(int i =0; i < 8; i++){
        cw_vehicle.platepoint_coord[i] = output_coordinates_mapped(valid_count, i) * plate_shape[i % 2];

        if(i % 2){
          cw_vehicle.platepoint_score[(i - 1) / 2] = output_confidences_mapped(valid_count, (i - 1) / 2);
        }
      }

      valid_count = valid_count + 1;

      // VLOG(0) << "platepoint_coord: " << cw_vehicle.platepoint_coord[0] << ", " << cw_vehicle.platepoint_coord[1] << ", " << cw_vehicle.platepoint_coord[2] << ", " << cw_vehicle.platepoint_coord[3]
              // << ", " << cw_vehicle.platepoint_coord[4] << ", " << cw_vehicle.platepoint_coord[5] << ", " << cw_vehicle.platepoint_coord[6] << ", " << cw_vehicle.platepoint_coord[7];
      // VLOG(0) << "platepoint_score: " << cw_vehicle.platepoint_score[0] << ", " << cw_vehicle.platepoint_score[1] << ", " << cw_vehicle.platepoint_score[2] << ", " << cw_vehicle.platepoint_score[3];    
    }  
  }

  return Status::OK();

}


TensorProto GetPlateRecognizeInput(const std::vector<cv::Mat>& origin_image_vec,
                                   std::vector<CW_VEHICLE>& vehicle_result_vec)
{
  // get image_vec
  std::vector<cv::Mat> image_vec;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    if(vehicle_result_vec[object_id].has_plate){
      CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];
      // filter by plate point
      if(cw_vehicle.platepoint_score[0] >= PLATE_POINT_THRESH &&
         cw_vehicle.platepoint_score[1] >= PLATE_POINT_THRESH &&
         cw_vehicle.platepoint_score[2] >= PLATE_POINT_THRESH &&
         cw_vehicle.platepoint_score[3] >= PLATE_POINT_THRESH)
      {
        // correct images, base on enlarge plate image
        float min_left = min(cw_vehicle.platepoint_coord[0], cw_vehicle.platepoint_coord[6]);
        float min_top = min(cw_vehicle.platepoint_coord[1], cw_vehicle.platepoint_coord[3]);
        float max_right = max(cw_vehicle.platepoint_coord[2], cw_vehicle.platepoint_coord[4]);
        float max_bottom = max(cw_vehicle.platepoint_coord[5], cw_vehicle.platepoint_coord[7]);
        float mid_width = (min_left + max_right) / 2.0;
        float mid_height = (min_top + max_bottom) / 2.0;
        float point_width = (sqrt(pow((cw_vehicle.platepoint_coord[2] - cw_vehicle.platepoint_coord[0]), 2) + pow((cw_vehicle.platepoint_coord[3] - cw_vehicle.platepoint_coord[1]), 2)) + 
                             sqrt(pow((cw_vehicle.platepoint_coord[4] - cw_vehicle.platepoint_coord[6]), 2) + pow((cw_vehicle.platepoint_coord[5] - cw_vehicle.platepoint_coord[7]), 2)))/ 2;
        float point_height = (sqrt(pow((cw_vehicle.platepoint_coord[6] - cw_vehicle.platepoint_coord[0]), 2) + pow((cw_vehicle.platepoint_coord[7] - cw_vehicle.platepoint_coord[1]), 2)) + 
                              sqrt(pow((cw_vehicle.platepoint_coord[4] - cw_vehicle.platepoint_coord[2]), 2) + pow((cw_vehicle.platepoint_coord[5] - cw_vehicle.platepoint_coord[3]), 2)))/ 2;
        
        // VLOG(0) << "point image: " << mid_width << ", " << mid_height << ", " << point_width << ", " << point_height;
        
        // src points for perspective transform
        cv::Point2f src_points[4] = { 
          cv::Point2f(cw_vehicle.platepoint_coord[0], cw_vehicle.platepoint_coord[1]),
          cv::Point2f(cw_vehicle.platepoint_coord[2], cw_vehicle.platepoint_coord[3]),
          cv::Point2f(cw_vehicle.platepoint_coord[4], cw_vehicle.platepoint_coord[5]),
          cv::Point2f(cw_vehicle.platepoint_coord[6], cw_vehicle.platepoint_coord[7]),
        };
        // dst points
        cv::Point2f dst_points[4] = { 
          cv::Point2f(mid_width - point_width / 2, mid_height - point_height / 2),
          cv::Point2f(mid_width + point_width / 2, mid_height - point_height / 2),
          cv::Point2f(mid_width + point_width / 2, mid_height + point_height / 2),
          cv::Point2f(mid_width - point_width / 2, mid_height + point_height / 2),
        };
        cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);

        // for(int i = 0; i < 4; i++){
        //   cv::Point2f point = src_points[i];
        //   VLOG(0) << "src_point: " << point.x << ", " << point.y;
        //   point = dst_points[i];
        //   VLOG(0) << "dst_point: " << point.x << ", " << point.y;
        // }
        // VLOG(0) << "transform_matrix: " << transform_matrix;

        // get correct image
        cv::Mat origin_image = origin_image_vec[cw_vehicle.batch_id].clone();
        cv::Mat plate_image = origin_image(cw_vehicle.enlarge_plate_rect);
        cv::Mat correct_image;
        cv::warpPerspective(plate_image, correct_image, transform_matrix, cv::Size(cw_vehicle.enlarge_plate_rect.width, cw_vehicle.enlarge_plate_rect.height));



        //get roi and roi_resize
        cv::Rect correct_image_rect;     
        correct_image_rect.x = max(0, floor(mid_width - point_width * (0.5 + PLATE_ENLARGE_WIDTH)));
        correct_image_rect.y = max(0, floor(mid_height - point_height * (0.5 + PLATE_ENLARGE_HEIGHT)));
        correct_image_rect.width = min(cw_vehicle.enlarge_plate_rect.width, floor(mid_width + point_width * (0.5 + PLATE_ENLARGE_WIDTH))) - correct_image_rect.x;
        correct_image_rect.height = min(cw_vehicle.enlarge_plate_rect.height, floor(mid_height + point_height * (0.5 + PLATE_ENLARGE_HEIGHT))) - correct_image_rect.y;
        cv::Mat roi = correct_image(correct_image_rect);
        // VLOG(0) << "correct_image_rect: " << correct_image_rect.x << ", " << correct_image_rect.y << ", " 
        //         << correct_image_rect.width + correct_image_rect.x << ", " << correct_image_rect.height + correct_image_rect.y;
        
        // resize    
        cv::Mat resize_roi;
        cv::resize(roi, resize_roi, cv::Size(PLATE_REC_WIDTH, PLATE_REC_HEIGHT));
        
        image_vec.push_back(resize_roi);
      }
      else{
        cw_vehicle.has_plate = false;
      }
    }          
  }

  // get input_tensor_proto
  TensorProto input_tensor_proto = imagevec2tensorproto(image_vec);

  return input_tensor_proto;

}


Status PostProcessPlateRecognize(const std::vector<cv::Mat>& origin_image_vec,
                                 std::vector<CW_VEHICLE>& vehicle_result_vec,
                                 const std::vector<string>& output_tensor_aliases,
                                 const std::vector<Tensor>& outputs)
{
  // check alias and tensor
  if (output_tensor_aliases.size() != outputs.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }

  // output of plate recognise
  Tensor output_confidences;
  Tensor output_indices;
  for(int i = 0; i < outputs.size(); i++){
    if(output_tensor_aliases[i].compare("output_confidences") == 0){
      output_confidences = outputs[i];
    }
    else if(output_tensor_aliases[i].compare("output_indices") == 0){
      output_indices = outputs[i];
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN,
             "Predict internal error, plate recognise output not found!");
    }
  }

  int sequence_length = output_indices.dim_size(1);

  auto output_indices_mapped = output_indices.tensor<int64, 2>();
  auto output_confidences_mapped = output_confidences.tensor<float, 2>();  

  int valid_count = 0;
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];
    if(vehicle_result_vec[object_id].has_plate){
      // get origin result
      int origin_indices[sequence_length];
      float origin_confidences[sequence_length];
      for(int i = 0; i < sequence_length; i++){
        origin_indices[i] = output_indices_mapped(valid_count, i);
        origin_confidences[i] = output_confidences_mapped(valid_count, i);
        // VLOG(0) << origin_indices[i];
        // VLOG(0) << origin_confidences[i];
      }
      valid_count = valid_count + 1;

      //decode
      //delete repeat
      std::vector<int> indices_without_repeat;
      std::vector<float> confidences_without_repeat;
      indices_without_repeat.push_back(origin_indices[0]);
      confidences_without_repeat.push_back(origin_confidences[0]);
      for(int i = 1; i < sequence_length; i++){
        if(origin_indices[i] == indices_without_repeat.back()){
          if(origin_confidences[i] > confidences_without_repeat.back()){
            confidences_without_repeat[confidences_without_repeat.size() - 1] = origin_confidences[i];
          }
        }
        else{
          indices_without_repeat.push_back(origin_indices[i]);
          confidences_without_repeat.push_back(origin_confidences[i]);
        }
      }
      //delete blank
      std::vector<int> indices_without_blank;
      std::vector<float> confidences_without_blank;
      for(int i = 0; i < indices_without_repeat.size(); i++){
        if(indices_without_repeat[i] != PLATE_CHAR_NUM){
          indices_without_blank.push_back(indices_without_repeat[i]);
          confidences_without_blank.push_back(confidences_without_repeat[i]);
        }
      }
      //province special and at the first position
      int province_position = -1;
      float province_probility = 0;
      for(int i = 0; i < indices_without_blank.size(); i++){
        if(indices_without_blank[i] < PLATE_PROVINCE_NUM && confidences_without_blank[i] > province_probility){
          province_position = i;
          province_probility = confidences_without_blank[i];
        }
      }
      if(province_position == -1){
        province_position = 0;
      }
      //get sub sequence
      int valid_char_count = 0;
      if(indices_without_blank.size() > 0){
        cw_vehicle.platerec_id[0] = indices_without_blank[province_position];
        cw_vehicle.platerec_score = confidences_without_blank[province_position];
        valid_char_count += 1;
        for(int i = province_position + 1; i < indices_without_blank.size(); i++){
          if(indices_without_blank[i] >= PLATE_PROVINCE_NUM){
            cw_vehicle.platerec_id[valid_char_count] = indices_without_blank[i];
            cw_vehicle.platerec_score *= confidences_without_blank[i];
            valid_char_count += 1;
          }
        }
      }
      // fill with -1 to rest id if valid_char_count >= 7
      if(valid_char_count < 7){
        for(int i = 0; i < PLATE_MAX_LENGTH; i++){
          cw_vehicle.platerec_id[i] = -1;
        }
        cw_vehicle.platerec_score = 0;
      }
      else{
        for(int i = valid_char_count; i < PLATE_MAX_LENGTH; i++){
          cw_vehicle.platerec_id[i] = -1;
        }
      }
      

      // VLOG(0) << "platerec_id: " << cw_vehicle.platerec_id[0] << ", " << cw_vehicle.platerec_id[1] << ", " << cw_vehicle.platerec_id[2] << ", " << cw_vehicle.platerec_id[3]
              // << ", " << cw_vehicle.platerec_id[4] << ", " << cw_vehicle.platerec_id[5] << ", " << cw_vehicle.platerec_id[6] << ", " << cw_vehicle.platerec_id[7];
      // VLOG(0) << "platerec_score: " << cw_vehicle.platerec_score;
    }
    else{
      // fill with -1 to object without plate
      for(int i = 0; i < PLATE_MAX_LENGTH; i++){
        cw_vehicle.platerec_id[i] = -1;
      }
    }
  }

  return Status::OK();

}



Status RunPredict(const string model_spec_name,
                  const string model_spec_signature_name,
                  const TensorProto& input_tensor_proto,
                  const std::vector<cv::Mat>& origin_image_vec,
                  std::vector<CW_VEHICLE>& vehicle_result_vec,
                  ServerCore* core,
                  const RunOptions& run_options,
                  PredictResponse* response)
{
  // prepare request
  PredictRequest request;
  request.mutable_model_spec()->set_name(model_spec_name);
  request.mutable_model_spec()->set_signature_name(model_spec_signature_name);
  google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& model_inputs = *request.mutable_inputs();
  model_inputs["input_images"] = input_tensor_proto;

  //debug input tensor proto
  // std::vector<cv::Mat> test_image_vec = tensorproto2imagevec(input_tensor_proto);
  // VLOG(0) << "input_images_shape: " << test_image_vec.size() << ", " << test_image_vec[0].cols << ", " << test_image_vec[0].rows << ", " << test_image_vec[0].channels();
  // cv::imwrite("input.jpg", test_image_vec[0]);

  // prepare bondle
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  const MetaGraphDef& meta_graph_def = bundle->meta_graph_def;
  const optional<int64>& servable_version = bundle.id().version; 
  Session* session = bundle->session.get();

  // Validate signatures.
  const string signature_name = request.model_spec().signature_name().empty()
                                    ? kDefaultServingSignatureDefKey
                                    : request.model_spec().signature_name();
  auto iter = meta_graph_def.signature_def().find(signature_name);
  if (iter == meta_graph_def.signature_def().end()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Serving signature key \"", signature_name, "\" not found."));
  }
  SignatureDef signature = iter->second;

  MakeModelSpec(request.model_spec().name(), signature_name, servable_version,
                response->mutable_model_spec());

  std::vector<std::pair<string, Tensor>> input_tensors;
  std::vector<string> output_tensor_names;
  std::vector<string> output_tensor_aliases;
  TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                          &output_tensor_names,
                                          &output_tensor_aliases));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(session->Run(run_options, input_tensors,
                                  output_tensor_names, {}, &outputs,
                                  &run_metadata));

  if(model_spec_name.compare("cw_common_detect_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessVehicleDetection(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else if(model_spec_name.compare("cw_car_color_type_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessVehicleColorType(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else if(model_spec_name.compare("cw_car_brand_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessVehicleBrand(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else if(model_spec_name.compare("cw_plate_detect_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessPlateDetection(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else if(model_spec_name.compare("cw_plate_key_point_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessPlatePoint(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else if(model_spec_name.compare("cw_plate_recognise_models") == 0){
    TF_RETURN_IF_ERROR(PostProcessPlateRecognize(origin_image_vec, vehicle_result_vec, output_tensor_aliases,outputs));
  }
  else{
      return tensorflow::Status(tensorflow::error::UNKNOWN, strings::StrCat(
        "run predict model error: ", model_spec_name));
  }

  return Status::OK();
}


void GetResponse(std::vector<CW_VEHICLE>& vehicle_result_vec,
                PredictResponse* response)
{
  // process type and brand, set brand if type in [0 2 3 4 5]
  for(int object_id = 0; object_id < vehicle_result_vec.size(); object_id++){
    CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];
    if(cw_vehicle.type_id >= 6 || cw_vehicle.type_id == 1){
      cw_vehicle.brand_id = -1;
      cw_vehicle.brand_score = 0;
    }
  }

  int object_count = vehicle_result_vec.size();
  // prepare data memory
  int *output_batch_id = (int *) malloc(sizeof(int) * object_count);
  memset(output_batch_id, -1, sizeof(int) * object_count);
  float *output_car_coord = (float *) malloc(sizeof(float) * object_count * 4);
  memset(output_car_coord, 0, sizeof(float) * object_count * 4);
  float *output_car_score = (float *) malloc(sizeof(float) * object_count);
  memset(output_car_score, 0, sizeof(float) * object_count);
  int *output_color_id = (int *) malloc(sizeof(int) * object_count);
  memset(output_color_id, -1, sizeof(int) * object_count);
  float *output_color_score = (float *) malloc(sizeof(float) * object_count);
  memset(output_color_score, 0, sizeof(float) * object_count);
  int *output_type_id = (int *) malloc(sizeof(int) * object_count);
  memset(output_type_id, -1, sizeof(int) * object_count);
  float *output_type_score = (float *) malloc(sizeof(float) * object_count);
  memset(output_type_score, 0, sizeof(float) * object_count);
  int *output_brand_id = (int *) malloc(sizeof(int) * object_count);
  memset(output_brand_id, -1, sizeof(int) * object_count);
  float *output_brand_score = (float *) malloc(sizeof(float) * object_count);
  memset(output_brand_score, 0, sizeof(float) * object_count);  
  int *output_platerec_id = (int *) malloc(sizeof(int) * object_count * PLATE_MAX_LENGTH);
  memset(output_platerec_id, -1, sizeof(int) * object_count  * PLATE_MAX_LENGTH);
  float *output_platerec_score = (float *) malloc(sizeof(float) * object_count);
  memset(output_platerec_score, 0, sizeof(float) * object_count);

  //prepare move data point
  int *mv_output_batch_id = output_batch_id;
  float *mv_output_car_coord = output_car_coord;
  float *mv_output_car_score = output_car_score;
  int *mv_output_color_id = output_color_id;
  float *mv_output_color_score = output_color_score;
  int *mv_output_type_id = output_type_id;
  float *mv_output_type_score = output_type_score;
  int *mv_output_brand_id = output_brand_id;
  float *mv_output_brand_score = output_brand_score;
  int *mv_output_platerec_id = output_platerec_id;
  float *mv_output_platerec_score = output_platerec_score;

  //get data value
  for(int object_id = 0; object_id < object_count; object_id++){
    const CW_VEHICLE& cw_vehicle = vehicle_result_vec[object_id];

    *mv_output_batch_id = cw_vehicle.batch_id;
    mv_output_batch_id++;
    memcpy(mv_output_car_coord, cw_vehicle.car_coord, sizeof(float) * 4);
    mv_output_car_coord = mv_output_car_coord + 4;
    *mv_output_car_score = cw_vehicle.car_score;
    mv_output_car_score++;
    *mv_output_color_id = cw_vehicle.color_id;
    mv_output_color_id++;
    *mv_output_color_score = cw_vehicle.color_score;
    mv_output_color_score++;
    *mv_output_type_id = cw_vehicle.type_id;
    mv_output_type_id++;
    *mv_output_type_score = cw_vehicle.type_score;
    mv_output_type_score++;
    *mv_output_brand_id = cw_vehicle.brand_id;
    mv_output_brand_id++;
    *mv_output_brand_score = cw_vehicle.brand_score;
    mv_output_brand_score++;
    memcpy(mv_output_platerec_id, cw_vehicle.platerec_id, sizeof(int) * PLATE_MAX_LENGTH);
    mv_output_platerec_id = mv_output_platerec_id + PLATE_MAX_LENGTH;
    *mv_output_platerec_score = cw_vehicle.platerec_score;
    mv_output_platerec_score++;
  }

  //prepare tensor
  Tensor batch_id_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({object_count}));
  Tensor car_coord_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count, 4}));
  Tensor car_score_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count}));
  Tensor color_id_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({object_count}));
  Tensor color_score_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count}));
  Tensor type_id_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({object_count}));
  Tensor type_score_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count}));
  Tensor brand_id_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({object_count}));
  Tensor brand_score_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count}));
  Tensor platerec_id_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({object_count, PLATE_MAX_LENGTH}));
  Tensor platerec_score_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({object_count}));

  //copy memory to tensor
  memcpy(batch_id_tensor.flat<int>().data(), output_batch_id, sizeof(int) * object_count);
  memcpy(car_coord_tensor.flat<float>().data(), output_car_coord, sizeof(float) * object_count * 4);
  memcpy(car_score_tensor.flat<float>().data(), output_car_score, sizeof(float) * object_count);
  memcpy(color_id_tensor.flat<int>().data(), output_color_id, sizeof(int) * object_count);
  memcpy(color_score_tensor.flat<float>().data(), output_color_score, sizeof(float) * object_count);
  memcpy(type_id_tensor.flat<int>().data(), output_type_id, sizeof(int) * object_count);
  memcpy(type_score_tensor.flat<float>().data(), output_type_score, sizeof(float) * object_count);
  memcpy(brand_id_tensor.flat<int>().data(), output_brand_id, sizeof(int) * object_count);
  memcpy(brand_score_tensor.flat<float>().data(), output_brand_score, sizeof(float) * object_count);
  memcpy(platerec_id_tensor.flat<int>().data(), output_platerec_id, sizeof(int) * object_count * PLATE_MAX_LENGTH);
  memcpy(platerec_score_tensor.flat<float>().data(), output_platerec_score, sizeof(float) * object_count);

  //set result to response
  batch_id_tensor.AsProtoField(&((*response->mutable_outputs())["output_batch_id"]));
  car_coord_tensor.AsProtoField(&((*response->mutable_outputs())["output_car_coord"]));
  car_score_tensor.AsProtoField(&((*response->mutable_outputs())["output_car_score"]));
  color_id_tensor.AsProtoField(&((*response->mutable_outputs())["output_color_id"]));
  color_score_tensor.AsProtoField(&((*response->mutable_outputs())["output_color_score"]));
  type_id_tensor.AsProtoField(&((*response->mutable_outputs())["output_type_id"]));
  type_score_tensor.AsProtoField(&((*response->mutable_outputs())["output_type_score"]));
  brand_id_tensor.AsProtoField(&((*response->mutable_outputs())["output_brand_id"]));
  brand_score_tensor.AsProtoField(&((*response->mutable_outputs())["output_brand_score"]));
  platerec_id_tensor.AsProtoField(&((*response->mutable_outputs())["output_platerec_id"]));
  platerec_score_tensor.AsProtoField(&((*response->mutable_outputs())["output_platerec_score"]));

  //release memory
  std::free(output_batch_id);
  std::free(output_car_coord);
  std::free(output_car_score);
  std::free(output_color_id);
  std::free(output_color_score);
  std::free(output_type_id);
  std::free(output_type_score);
  std::free(output_brand_id);
  std::free(output_brand_score);
  std::free(output_platerec_id);
  std::free(output_platerec_score);
  output_batch_id = NULL;
  output_car_coord = NULL;
  output_car_score = NULL;
  output_color_id = NULL;
  output_color_score = NULL;
  output_type_id = NULL;
  output_type_score = NULL;
  output_brand_id = NULL;
  output_brand_score = NULL;
  output_platerec_id = NULL;
  output_platerec_score = NULL;
  mv_output_batch_id = NULL;
  mv_output_car_coord = NULL;
  mv_output_car_score = NULL;
  mv_output_color_id = NULL;
  mv_output_color_score = NULL;
  mv_output_type_id = NULL;
  mv_output_type_score = NULL;
  mv_output_brand_id = NULL;
  mv_output_brand_score = NULL;
  mv_output_platerec_id = NULL;
  mv_output_platerec_score = NULL;

}


Status RunPredictVehicle(const RunOptions& run_options,
                         ServerCore* core,
                         const PredictRequest& request, PredictResponse* response) 
{
  // validate model_spec_name and signature_name
  const string model_spec_name = request.model_spec().name();
  if(model_spec_name.compare("cw_common_detect_models") !=0){
    return errors::FailedPrecondition(strings::StrCat(
        "Serving model_spec_name \"", model_spec_name, "\" not found."));
  }
  const string signature_name = request.model_spec().signature_name().empty()
                                  ? kDefaultServingSignatureDefKey
                                  : request.model_spec().signature_name();
  if(signature_name.compare("common_detect") != 0){
    return errors::FailedPrecondition(strings::StrCat(
        "Serving signature key \"", signature_name, "\" not found."));
  }

  VLOG(0) << "predict vehicle!";
  std::vector<CW_VEHICLE> vehicle_result_vec;

  // get input_tensor_proto from origin request
  TensorProto input_tensor_proto;
  for (auto& input : request.inputs()) {
    const string& input_alias = input.first;

    if(input_alias.compare("input_images") == 0){
      input_tensor_proto = input.second;
    }
  }

  // get origin_image_vec from input_tensor_proto
  std::vector<cv::Mat> origin_image_vec = tensorproto2imagevec(input_tensor_proto);

  // vehicle detection
  VLOG(0) << "vehicle detection!";
  // run vehicle detection predict
  TF_RETURN_IF_ERROR(RunPredict("cw_common_detect_models",
                                "common_detect",
                                input_tensor_proto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));

  // plate detection
  VLOG(0) << "plate detection!";
  // get input_tensor_proto
  TensorProto platedet_input_tensorproto = GetPlateDetectInput(origin_image_vec, vehicle_result_vec);
  //run plate detection
  TF_RETURN_IF_ERROR(RunPredict("cw_plate_detect_models",
                                "plate_detect",
                                platedet_input_tensorproto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));


  // plate point
  VLOG(0) << "plate point!";
  // get input_tensor_proto
  TensorProto platepoint_input_tensorproto = GetPlatePointInput(origin_image_vec, vehicle_result_vec);
  //run plate point
  TF_RETURN_IF_ERROR(RunPredict("cw_plate_key_point_models",
                                "plate_point",
                                platepoint_input_tensorproto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));

  // plate recognize
  VLOG(0) << "plate recognize!";
  // get input_tensor_proto
  TensorProto platerecognize_input_tensorproto = GetPlateRecognizeInput(origin_image_vec, vehicle_result_vec);
  //run plate recognize
  TF_RETURN_IF_ERROR(RunPredict("cw_plate_recognise_models",
                                "plate_recognize",
                                platerecognize_input_tensorproto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));

  // vehicle color and type
  VLOG(0) << "vehicle color and type!";
  // get input_tensor_proto
  TensorProto colortype_input_tensorproto = GetVehicleColorTypeInput(origin_image_vec, vehicle_result_vec);
  // run color and type predict
  TF_RETURN_IF_ERROR(RunPredict("cw_car_color_type_models",
                                "car_colortype",
                                colortype_input_tensorproto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));

  // vehicle brand
  VLOG(0) << "vehicle brand!";
  // get input_tensor_proto
  TensorProto brand_input_tensorproto = GetVehicleBrandInput(origin_image_vec, vehicle_result_vec);
  // run brand predict
  TF_RETURN_IF_ERROR(RunPredict("cw_car_brand_models",
                                "car_recognize",
                                colortype_input_tensorproto,
                                origin_image_vec, vehicle_result_vec, core, run_options, response));

  GetResponse(vehicle_result_vec, response);

  return Status::OK();

}

}  // namespace serving
}  // namespace tensorflow
