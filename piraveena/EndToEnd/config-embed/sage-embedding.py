import config.config as config


ACCESS_KEY = config.aws_keys['access_key_id']
SECRET_KEY = config.aws_keys['secret_access_key']
BUCKET_NAME = config.bucket_name
JOB_NAME=""
Iteration="iteration-1"


import boto3

import boto3
import os


def upload_files(folder_path,upload_path):
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=BUCKET_NAME)
    response = s3.list_buckets()

    # Get a list of all bucket names from the response
    buckets = [bucket['Name'] for bucket in response['Buckets']]

    # Print out the bucket list
    print("Bucket List: %s" % buckets)
    s3.upload_file(folder_path, BUCKET_NAME, upload_path, ExtraArgs={'ACL':'public-read'})



import re
from sagemaker import get_execution_role
import boto3
from time import gmtime, strftime
import time
import numpy as np
import os
import json
from validate import validate_params
import logging
from datetime import datetime

logging.getLogger('botocore').setLevel(logging.DEBUG)
logging.getLogger('boto3').setLevel(logging.DEBUG)


def embed(bucket,prefix,output):
    # role=get_execution_role()
    role = "arn:aws:iam::021301526575:role/sagemaker@piraveena"
    print(role)
    region_name = boto3.Session().region_name
    print(region_name)

    containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/blazingtext:latest',
                  'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/blazingtext:latest',
                  'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/blazingtext:latest',
                  'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/blazingtext:latest'}

    container = containers[region_name]
    print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))

    resource_config = {
            "InstanceCount": 2,
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 2
        }

    hyperparameters = {
            "mode": "batch_skipgram",
            "epochs": "5",
            "min_count": "10",
            "sampling_threshold": "0.0001",
            "learning_rate": "0.025",
            "window_size": "5",
            "vector_dim": "100",
            "negative_samples": "5",
            "batch_size": "11", #  = (2*window_size + 1) (Preferred)
            "evaluation": "true" # Perform similarity evaluation on WS-353 dataset at the end of training
        }


    validate_params(resource_config, hyperparameters)


    job_name = "DEMO-BT-text8-{}-{}-{}-".format(resource_config["InstanceCount"],
                                                resource_config["InstanceType"].replace(".","-"),
                                                hyperparameters["mode"].replace("_","-"))\
                                        + strftime("%Y-%m-%d-%H-%M", gmtime())
    global JOB_NAME
    JOB_NAME= job_name
    print("Training job", job_name)


    create_training_params = \
    {
        "TrainingJobName": job_name,
        "ResourceConfig": resource_config,
        "HyperParameters": hyperparameters,
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": "s3://{}/{}/".format(bucket, output)
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600 * 24 #Hours
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated"  # Always keep FullyReplicated for BlazingText
                    }
                },
            },
        ]
    }
    startTime = datetime.now()
    sagemaker_client = boto3.Session().client(service_name='sagemaker')
    sagemaker_client.create_training_job(**create_training_params)
    status = sagemaker_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(status)

    time.sleep(5)
    sagemaker_client.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)

    status = sagemaker_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(status)

    # if the job failed, determine why
    if status == 'Failed':
        message = sagemaker_client.describe_training_job(TrainingJobName=job_name)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')


    endtime=datetime.now()
    print("Duration : ",(endtime-startTime))

import botocore
def download(folder_path,download_path):

    s3 = boto3.resource('s3')
    try:
        s3.Bucket(BUCKET_NAME).download_file(folder_path, download_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise




if __name__ == "__main__":
    upload_files("/home/piraveena/Documents/Semi8/fyp-polysemy-embedding/data/histogram/text8",Iteration+"/file/data.txt")
    embed("ml-embedding",Iteration+"/file",Iteration+"/model")
    download(Iteration+"/model/"+JOB_NAME+"/output/model.tar.gz","/home/piraveena/Documents/Semi8/fyp-polysemy-embedding/data/histogram/model2.tar")
    #

