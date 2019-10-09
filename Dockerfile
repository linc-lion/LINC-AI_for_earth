# Pull in the AI for Earth Base Image, so we can extract necessary libraries.
FROM mcr.microsoft.com/aiforearth/base-py:1.8-cuda-9.0-runtime

# Copy requirements
COPY ./requirements.txt /

RUN echo "source activate ai4e_py_api" >> ~/.bashrc 
#RUN conda install -c conda-forge -n ai4e_py_api  
RUN /usr/local/envs/ai4e_py_api/bin/pip install --upgrade pip
RUN /usr/local/envs/ai4e_py_api/bin/pip install -r /requirements.txt

# Note: supervisor.conf reflects the location and name of your api code.
COPY ./supervisord.conf /etc/supervisord.conf

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

# Copy your API code
COPY ./Linc_deploy /app/Linc_deploy/

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    TRACE_SAMPLING_RATE=1.0 \ 
# The following variables will allow you to filter logs in AppInsights \ 
    SERVICE_OWNER=AI4E_Test \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=LINC\ API \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    ENVSERVICE_MODEL_VERSION=1.0 \
    API_PREFIX=/LINC \
    LION_MODEL_PATH=/app/Linc_deploy/Models/body_parts.pth \
	WHISKER_MODEL_PATH=/app/Linc_deploy/Models/whiskers.pth \ 
    MODEL_VERSION=fasterrcnn_resnet50_fpn \ 
    MAX_IMAGES_ACCEPTED=2 \ 
    GPU_BATCH_SIZE=8 \ 
    DEFAULT_DETECTION_CONFIDENCE=0.7


# Expose the port that is to be used when calling your API
EXPOSE 3003
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost:3003${API_PREFIX}  | echo || exit 1
ENTRYPOINT [ "/startup.sh" ]
