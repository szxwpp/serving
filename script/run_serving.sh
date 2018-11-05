cd /serving

cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server . 

CUDA_VISIBLE_DEVICES=3  ./tensorflow_model_server --port=8888 --model_config_file=./config/model_config_vehicle.cfg --platform_config_file=./config/platform_config_file.cfg
