import os

import boto
import boto.s3

def upload():
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    file_path = path + file
    print('Uploading %s to Amazon S3 bucket %s' % (file_path, bucket_name))
    bucket = conn.get_bucket(bucket_name)

    full_key_name = os.path.join(s3_path, file)
    print("s3 path: ", full_key_name)
    k = bucket.new_key(full_key_name)
    print("Start uploading to ", full_key_name)
    k.set_contents_from_filename(file_path)
    print("Finished uploading")


if __name__ == '__main__':
    AWS_ACCESS_KEY_ID = ''
    AWS_SECRET_ACCESS_KEY = ''
    file = "text8.zip"  #Name of the file to be uploaded
    path= "/home/piraveena/fyp/"     #Path of the file to be uploaded
    bucket_name = 'polysemy-embedding'  #S3 bucket name
    s3_path = '/polysemy/data/'     #Directory Under which file should get upload

    upload()

