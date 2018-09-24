import re
# from sagemaker import get_execution_role
import boto3
from time import gmtime, strftime
import time
import numpy as np
import os
import json
from PolysemyEmbedding.SupportModules import SageMakerValidator
import logging
from datetime import datetime

logging.getLogger('botocore').setLevel(logging.DEBUG)
logging.getLogger('boto3').setLevel(logging.DEBUG)



bucket= "word-embeding-data"
prefix='polysemyEmbedding/graphclustering/iteration_7_1/data'
output='polysemyEmbedding/graphclustering/iteration_7_1'
# prefix ='test/data'
# output = 'test'
# a,h=0
# #
# # def upload_to_s3(bucket, prefix, channel, file):
# #     s3 = boto3.resource('s3')
# #     data = open(file, "rb")
# #     key = prefix + "/" + channel + '/' + file
# #     s3.Bucket(bucket).put_object(Key=key, Body=data)
# #
# # upload_to_s3(bucket, prefix, 'train', 'text8')
#
#
#
#
#
#
# role=get_execution_role()
#
# client = boto3.client(
#     's3',
#     aws_access_key_id='AKIAIZKEEZRAOYZAJWPQ',
#     aws_secret_access_key='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',
#     region_name='us-west-1'
# )
session = boto3.Session(
    aws_access_key_id='AKIAIZKEEZRAOYZAJWPQ',
    aws_secret_access_key='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',
    region_name='us-west-2'
)
s3 = session.resource('s3')

role = "role"
print(role)
region_name = session
print(region_name)

containers = {'us-west-2':
'433757028032.dkr.ecr.us-west-2.amazonaws.com/blazingtext:latest',
              'us-east-1':
'811284229777.dkr.ecr.us-east-1.amazonaws.com/blazingtext:latest',
              'us-east-2':
'825641698319.dkr.ecr.us-east-2.amazonaws.com/blazingtext:latest',
              'eu-west-1':
'685385470294.dkr.ecr.eu-west-1.amazonaws.com/blazingtext:latest'}

container = containers[region_name]
print('Using SageMaker BlazingText container: {}({})'.format(container, region_name))

resource_config = {
        "InstanceCount": 1,
        "InstanceType": "ml.p3.2xlarge",
        "VolumeSizeInGB": 10
    }

hyperparameters = {
        "mode": "skipgram",
        "epochs": "5",
        "min_count": "15    ",
        "sampling_threshold": "0.0001",
        "learning_rate": "0.025",
        "window_size": "5",
        "vector_dim": "100",
        "negative_samples": "5",
        "batch_size": "11", #  = (2*window_size + 1) (Preferred)
        "evaluation": "true" # Perform similarity evaluation on WS-353dataset at the end of training
    }


SageMakerValidator.validate_params(resource_config, hyperparameters)


job_name = "DEMO-BT-text8-{}-{}-{}-".format(resource_config["InstanceCount"],

resource_config["InstanceType"].replace(".","-"),

hyperparameters["mode"].replace("_","-"))\
                                    + strftime("%Y-%m-%d-%H-%M", gmtime())
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
                    "S3DataDistributionType": "FullyReplicated"  #Always keep FullyReplicated for BlazingText
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


# endtime=datetime.now()
# print("Duration : ",(endtime-startTime))
#
# info = sagemaker_client.describe_training_job(TrainingJobName=job_name)
# if "ModelArtifacts" not in info:
#     raise Exception('Could not find model artifacts. Please wait forthe job to finish!')
# key = "/{}/output/model.tar.gz".format(info["TrainingJobName"])
# s3 = boto3.resource('s3')
# s3.Bucket(bucket).download_file(key, 'model.tar.gz')


# import numpy as np
# from sklearn.preprocessing import normalize
#
# # Read the 400 most frequent word vectors. The vectors in the fileare in descending order of frequency.
# num_points = 500
#
# first_line = True
# index_to_word = []
# map={}
# with open("/home/singam/Documents/fyp/low-resource-embedding/tamil/model(17)/vectors.txt","r")as f:
#     for line_num, line in enumerate(f):
#         if first_line:
#             dim = int(line.strip().split()[1])
#             word_vecs = np.zeros((num_points, dim), dtype=float)
#             first_line = False
#             continue
#         line = line.strip()
#         word = line.split()[0]
#         vec = word_vecs[line_num-1]
#         for index, vec_val in enumerate(line.split()[1:]):
#             vec[index] = float(vec_val)
#         map[word]=vec
#         index_to_word.append(word)
#         if line_num >= num_points:
#             break
# word_vecs = normalize(word_vecs, copy=False, return_norm=False)
# from sklearn.manifold import TSNE
#
# tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
# vecs=word_vecs[:num_points]
#
# two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
# labels = index_to_word[:num_points]
# from matplotlib import pylab
#
#
# def plot(embeddings, labels):
#     pylab.figure(figsize=(20,20))
#     for i, label in enumerate(labels):
#         x, y = embeddings[i,:]
#         pylab.scatter(x, y)
#         pylab.annotate(label, xy=(x, y), xytext=(5, 2),textcoords='offset points',
#                        ha='right', va='bottom')
#
#     pylab.savefig("result.png", format='png');
#     pylab.show();
#
#
# plot(two_d_embeddings, labels)

