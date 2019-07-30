# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mcr.microsoft.com/aiforearth/base-py:1.8-cuda-9.0-runtime

# Note: supervisor.conf reflects the location and name of your api code.
COPY ./supervisord.conf /etc/supervisord.conf
# startup.sh is a helper script
COPY ./startup.sh /
# Copy your API code
COPY ./Linc_deploy /app/Linc_deploy/

RUN echo "source activate ai4e_py_api" >> ~/.bashrc \
    && conda install -c conda-forge -n ai4e_py_api numpy pandas && \
    /usr/local/envs/ai4e_py_api/bin/pip install --upgrade pip && \
    /usr/local/envs/ai4e_py_api/bin/pip install tensorflow==1.9 pillow requests_toolbelt torchvision==0.3.0 && \
    chmod +x /startup.sh

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    TRACE_SAMPLING_RATE=1.0 \ 
# The following variables will allow you to filter logs in AppInsights \ 
    SERVICE_OWNER=AI4E_Test \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=base-py\ example \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    ENVSERVICE_MODEL_VERSION=1.0 \
    API_PREFIX=/LINC_V1/Linc_deploy/ \
    MODEL_PATH=/app/Linc_deploy/Model/body_parts_0_0_1.pth \
    MODEL_VERSION=fasterrcnn_resnet50_fpn \ 
    MAX_IMAGES_ACCEPTED=3 \ 
    GPU_BATCH_SIZE=8 \ 
    DEFAULT_DETECTION_CONFIDENCE=0.5

# Expose the port that is to be used when calling your API
EXPOSE 1212
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost:1212/${API_PREFIX}/  || exit 1
ENTRYPOINT [ "/startup.sh" ]
