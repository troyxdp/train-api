FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./yolov7 /code/yolov7
COPY ./yolov9 /code/yolov9
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# Update package list and install necessary packages
# RUN apt-get update && \
#     apt-get install -y \
#         zip \
#         htop \
#         screen \
#         libgl1-mesa-glx && \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y zip
RUN apt-get install -y htop
RUN apt-get install -y screen
RUN apt-get install -y libgl1-mesa-glx
RUN rm -rf /var/lib/apt/lists/*
# Install Python packages
RUN pip install seaborn thop
RUN pip install nvidia-pyindex
RUN pip install onnx-graphsurgeon
RUN python3 -m pip install colored
RUN pip install onnxruntime
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]