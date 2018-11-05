docker run -it -p 8888:8888 --runtime=nvidia \
--name tfserving \
--mount type=bind,source=/home/shizhixiang/workspace/serving,target=/serving \
tensorflow/serving:1.10.0-devel-gpu-v1 /bin/bash
