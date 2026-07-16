import io
import os
import boto3
import joblib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

### -------- Constants
BUCKET_NAME = "halfmarathon-test"
S3_SOURCE_FILES = "halfmarathons"
S3_MOD_FILE = "Models"
S3_CLEANED_FILES = "cleaned_HF_files"
# S3_MODEL_NAME = "best_5km_model_2023_2024"

### -------------------------------------------- ###
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
    
)


@st.cache_resource
def load_model(file: str):
    file_path = os.path.join(S3_MOD_FILE, file).replace("\\","/")
    file_path = f'{file_path}.pkl'
    
    try:
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=file_path
        )
        model_bytes = response['Body'].read()
        model = joblib.load(io.BytesIO(model_bytes))
        return model
    except Exception as e:
        st.error(f"Nie udało się pobrać modelu z chmury: {e}")
        return None

    # print(f"Wczytano plik {file}")