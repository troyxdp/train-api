#! /usr/bin/bash
sudo docker stop train-api
sudo docker rm train-api
sudo docker rmi train-api
cd training
sudo rm -r rf_intrusion_yolov7x_v4
sudo rm -r rf_intrusion_yolov9_v1
cd ..
sudo docker build -t train-api .
sudo docker run -d --name train-api --runtime=nvidia --gpus all -p 5431:80 -v /home/refraime/Documents/Troy/train-api:/code train-api
sudo docker logs --follow train-api