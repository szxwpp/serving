/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

#include <string>
#include <utility>
#include <sys/time.h>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

#include "predict_util_vehicle.h"

namespace tensorflow {
namespace serving {

namespace {

Status SessionBundlePredict(const RunOptions& run_options,
                            const MetaGraphDef& meta_graph_def,
                            const optional<int64>& servable_version,
                            const PredictRequest& request,
                            PredictResponse* response, Session* session) {
  // Validate signatures.
  Signature signature;
  TF_RETURN_IF_ERROR(GetNamedSignature("inputs", meta_graph_def, &signature));
  if (!signature.has_generic_signature()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "'inputs' named signature is not a generic signature");
  }
  GenericSignature input_signature = signature.generic_signature();
  TF_RETURN_IF_ERROR(GetNamedSignature("outputs", meta_graph_def, &signature));
  if (!signature.has_generic_signature()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "'outputs' named signature is not a generic signature");
  }
  GenericSignature output_signature = signature.generic_signature();

  // Verify and prepare input.
  if (request.inputs().size() != input_signature.map().size()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "input size does not match signature");
  }
  std::vector<std::pair<string, Tensor>> inputs;
  for (auto& input : request.inputs()) {
    const string& alias = input.first;
    auto iter = input_signature.map().find(alias);
    if (iter == input_signature.map().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          "input tensor alias not found in signature: " + alias);
    }
    Tensor tensor;
    if (!tensor.FromProto(input.second)) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "tensor parsing error: " + alias);
    }
    inputs.emplace_back(std::make_pair(iter->second.tensor_name(), tensor));
  }

  // Prepare run target.
  std::set<string> seen_outputs;
  std::vector<string> output_filter(request.output_filter().begin(),
                                    request.output_filter().end());
  std::vector<string> output_tensor_names;
  std::vector<string> output_aliases;
  for (auto& alias : output_filter) {
    auto iter = output_signature.map().find(alias);
    if (iter == output_signature.map().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          "output tensor alias not found in signature: " + alias);
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "duplicate output tensor alias: " + alias);
    }
    seen_outputs.insert(alias);
    output_tensor_names.emplace_back(iter->second.tensor_name());
    output_aliases.emplace_back(alias);
  }
  // When no output is specified, fetch all output tensors specified in
  // the signature.
  if (output_tensor_names.empty()) {
    for (auto& iter : output_signature.map()) {
      output_tensor_names.emplace_back(iter.second.tensor_name());
      output_aliases.emplace_back(iter.first);
    }
  }

  MakeModelSpec(request.model_spec().name(), /*signature_name=*/{},
                servable_version, response->mutable_model_spec());

  // Run session.
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(session->Run(run_options, inputs, output_tensor_names, {},
                                  &outputs, &run_metadata));

  // Validate and return output.
  if (outputs.size() != output_tensor_names.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i].AsProtoField(
        &((*response->mutable_outputs())[output_aliases[i]]));
  }

  return Status::OK();
}

}  // namespace

Status TensorflowPredictor::Predict(const RunOptions& run_options,
                                    ServerCore* core,
                                    const PredictRequest& request,
                                    PredictResponse* response) {

  bool use_saved_model_cascade= false;
  int cascade_flag = -1;
  for (auto& input : request.inputs()){
    const string& input_alias = input.first;
    VLOG(0) << "request.inputs: " << input_alias;

    if (input_alias.compare("cascade_flag") == 0){
      use_saved_model_cascade = true;
      Tensor tensor_cascade_flag;
      tensor_cascade_flag.FromProto(input.second);
      auto value_cascade_flag = tensor_cascade_flag.tensor<int, 1>();
      cascade_flag = value_cascade_flag(0);
    }
  }

  // predict cascade models
  if (use_saved_model_cascade){
    struct timeval tic, toc;    // test speed
    gettimeofday(&tic, NULL);   // start
    
    // vehicle
    if (cascade_flag == 1){
      TF_RETURN_IF_ERROR(RunPredictVehicle(run_options, core, request, response));
    }
    else{
      return tensorflow::Status(tensorflow::error::UNKNOWN, "cascade mode is not supported");
    }

    
    // test time
    gettimeofday(&toc, NULL);   // end
    VLOG(0) << "predict once Elapse: " << ((toc.tv_sec - tic.tv_sec) * 1000000 + (toc.tv_usec - tic.tv_usec)) / 1000 << " ms"; // test speed

    return Status::OK();
  }

  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }

  if (use_saved_model_) {
    ServableHandle<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
    return RunPredict(run_options, bundle->meta_graph_def, bundle.id().version,
                      bundle->session.get(), request, response);
  }
  ServableHandle<SessionBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  return SessionBundlePredict(run_options, bundle->meta_graph_def,
                              bundle.id().version, request, response,
                              bundle->session.get());
}

}  // namespace serving
}  // namespace tensorflow
