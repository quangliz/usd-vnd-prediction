import os 
from dotenv import load_dotenv
from joblib import load
import boto3
import pandas as pd
import io

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH_S3")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH_S3")
S3_BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("REGION")

def load_data_from_s3():
    print("Loading data from S3")
    s3 = boto3.client('s3', region_name=REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key=DATA_PATH)
    file_content = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(file_content))
    df.set_index('Ngày', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def load_model_from_s3():
    print("Loading model from S3")
    s3 = boto3.client('s3', region_name=REGION)
    obj = s3.get_object(Bucket=S3_BUCKET, Key=DEFAULT_MODEL_PATH)
    # print(obj)
    model = load(io.BytesIO(obj['Body'].read()))
    # print(model)
    return model