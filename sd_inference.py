import json
import boto3
from PIL import Image
import base64
import io

# Create one from AWS Sagemaker pretrained models
# Look something like this "jumpstart-dft-sd-2-inpainting-fp16-20250319-115544"
Inpaint_aws_endpoint = ""

def image_to_byte_array(image: Image, format: str = 'PNG'):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def encode_img(image: Image, format: str = 'PNG'):
    img_bytes = image_to_byte_array(image, format=format)
    encoded_img = base64.b64encode(bytearray(img_bytes)).decode()
    return encoded_img



def call_sagemaker_inpaint(payload):
    """query the endpoint with the json payload encoded in utf-8 format."""
    encoded_payload = json.dumps(payload).encode('utf-8')
    client = boto3.client('runtime.sagemaker')
    # Accept = 'application/json;jpeg' returns the jpeg image as bytes encoded by base64.b64 encoding.
    # To receive raw image with rgb value set Accept = 'application/json'
    # To send raw image, you can set content_type = 'application/json' and encoded_image as np.array(PIL.Image.open('low_res_image.jpg')).tolist()
    # Note that sending or receiving payload with raw/rgb values may hit default limits for the input payload and the response size.
    response = client.invoke_endpoint(EndpointName=Inpaint_aws_endpoint, ContentType='application/json;jpeg', Accept = 'application/json;jpeg', Body=encoded_payload)
    return response