# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
import json
from flask import Flask, request, abort
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
from os import getenv
from datetime import datetime
from tempfile import SpooledTemporaryFile
from predict_AI import LINC_detector

print('Creating Application')
app = Flask(__name__)

# Use the AI4EAppInsights library to send log messages. NOT REQURIED
log = AI4EAppInsights()

# Use the APIService to execute your functions within a logging trace, which supports long-running/async functions,
# handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)

# Load LINC models
cuda_support = getenv('CUDA_SUPPORT')   # Either CUDA or normal cpu
LINC_Lion = LINC_detector(getenv('LION_MODEL_PATH'),cpu=cuda_support)
LINC_Whisker = LINC_detector(getenv('WHISKER_MODEL_PATH'),cpu=cuda_support)
print("LABEL_NAMES",LINC_Lion.label_names)

#Helper
class File_Error(Exception):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}

class Param_Error(Exception):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}


# Define a function for processing request data,
# This function is passed as a param to API setup.
# Return_values passed to **kwargs
def process_request_data(requests):
    image_error = """
    File count error. 
    Maximum image count: {},
    Min image count 1
    """.format(str(getenv('MAX_IMAGES_ACCEPTED')))

    conf_error = """
    Detection confidence needs 
    to be between 0.0 and 1.0."""

    return_values = {'images': None}
    try:
        # Check 'conf'(confidence) param
        args = requests.args
        if 'conf' in args: 
            detection_confidence = float(args['conf'])
            if detection_confidence < 0.0 or detection_confidence > 1.0:
                raise Param_Error("Bad conf number")# Throw 400 bad parameter
            else:
                return_values['conf'] = detection_confidence 
        else:
            return_values['conf'] = float(getenv('DEFAULT_DETECTION_CONFIDENCE'))
        
        # Check num images
        temp_files = []
        temp_names = []
        num_files = len(requests.files)
        max_img = int(getenv('MAX_IMAGES_ACCEPTED'))
        print('MaxImg:', getenv('MAX_IMAGES_ACCEPTED'))
        print('NumImages:', len(requests.files))
        if num_files > max_img or num_files < 1: # Throw 413 too many files
            raise File_Error("File count error")
        for f_key, the_file in requests.files.items():
            fType = the_file.filename.lower()
            if not fType.endswith(('.jpeg','.jpg','.png')):
                print('file error')             # Throw 415 format error
                abort(415,"File error, .jpg, .jpeg or .png only")
            # Make anon-file 
            temp = SpooledTemporaryFile()
            temp.write(request.files[f_key].read())
            temp_files.append(temp)
            temp_names.append(f_key)
 
        return_values['images'] = temp_files
        return_values['inames'] = temp_names

        #DEBUG
        for key,value in args.items():
            print("Param:{}={}".format(key,value))
        n_files = len(requests.files)       
        print("number_files:",n_files)
        print("conf: ", return_values['conf'])
    
    except Param_Error:
        abort(400,conf_error)
        log.log_error('Unable to load the request data')   # Log to Application Insights
    except File_Error:
        abort(413, image_error)
        log.log_error('Unable to load the request data')   # Log to Application Insights
    return return_values    


# POST, Blocking
@ai4e_service.api_sync_func(
    api_path = '/detect_lion', 
    methods = ['POST'], 
    content_types = ['image/png', 'application/octet-stream', 'image/jpeg'],
    #content_max_length = 5 * 8 * 1000000,  # 5MB per image * number of images allowed
    request_processing_function = process_request_data, # This is the data process function that you created above.
    maximum_concurrent_requests = 5, 
    trace_name = 'post:detect_lion')
def get_detect_lion(*args, **kwargs):
    # Get data from process_request_data
    image_files = kwargs.get('images')
    image_names = kwargs.get('inames')
    detection_confidence = kwargs.get('conf')
    print("Recieved data:")
    print(image_files, image_names, detection_confidence)
    try:
        print('runserver, post_detect_sync, batching and inferencing...')
        tic = datetime.now()
        result = LINC_Lion.detect(image_files, image_names, detection_confidence)
        toc = datetime.now()
        
        inference_duration = toc - tic
        print('runserver, post_detect_sync, inference duration: {} seconds.'.format(inference_duration))
    except Exception as e:
        print('Error performing detection on the images: ' + str(e))
        log.log_exception('Error performing detection on the images: ' + str(e))
        return -1 
    return json.dumps(result)

# POST, Blocking
@ai4e_service.api_sync_func(
    api_path = '/detect_whisker', 
    methods = ['POST'], 
    maximum_concurrent_requests = 5, 
    content_types = ['image/png', 'application/octet-stream', 'image/jpeg'],
    content_max_length = 5 * 8 * 1000000,  # 5MB per image * number of images allowed
    request_processing_function = process_request_data, # This is the data process function that you created above.
    trace_name = 'get:get_classes')
def detect_whisker(*args, **kwargs):
    image_files = kwargs.get('images')
    image_names = kwargs.get('inames')
    detection_confidence = kwargs.get('conf')
    print("Recieved data:")
    print(image_files, image_names, detection_confidence)
    try:
        print('runserver, post_detect_sync, batching and inferencing...')
        tic = datetime.now()
        result = LINC_Whisker.detect(image_files, image_names, detection_confidence)
        toc = datetime.now()
        
        inference_duration = toc - tic
        print('runserver, post_detect_sync, inference duration: {} seconds.'.format(inference_duration))
    except Exception as e:
        print('Error performing detection on the images: ' + str(e))
        log.log_exception('Error performing detection on the images: ' + str(e))
        return -1 
    return json.dumps(result)


@ai4e_service.api_sync_func(
    api_path = '/classes', 
    methods = ['GET'],
    maximum_concurrent_requests = 1000, 
    trace_name = 'get:classes')
def get_classes(*args, **kwargs):
    classes = {
        "1":"cv-dl",
        "2":"cv-dr",
        "3":"cv-f",
        "4":"cv-sl",
        "5":"cv-sr",
        "6":"ear-dl-l",
        "7":"ear-dl-r",
        "8":"ear-dr-l",
        "9":"ear-dr-r",
        "10":"ear-fl",
        "11":"ear-fr",
        "12":"ear-sl",
        "13":"ear-sr",
        "14":"eye-dl-l",
        "15":"eye-dl-r",
        "16":"eye-dr-l",
        "17":"eye-dr-r",
        "18":"eye-fl",
        "19":"eye-fr",
        "20":"eye-sl",
        "21":"eye-sr",
        "22":"nose-dl",
        "23":"nose-dr",
        "24":"nose-f",
        "25":"nose-sl",
        "26":"nose-sr",
        "27":"whisker-dl",
        "28":"whisker-dr",
        "29":"whisker-f",
        "30":"whisker-sl",
        "31":"whisker-sr",
    }
    return json.dumps(classes)

if __name__ == '__main__':
    app.run()
