"""Inference script for medical image classification with local file support."""

import json
import logging
import os
import base64
import tempfile
from typing import Any, Dict, Union
import torch
import boto3
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def s3_uri_to_bucket_key(s3_uri: str) -> tuple:
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

def model_fn(model_dir: str) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=8)

    model_path = os.path.join(model_dir, "model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully")

    model.to(device).eval()

    transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=(256, 256, 1)),
        ScaleIntensity()
    ])

    return {"model": model, "device": device, "transforms": transforms}

def input_fn(request_body: Union[str, bytes], content_type: str) -> Any:
    if content_type == "application/dicom":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            temp_file.write(request_body)
            temp_path = temp_file.name
        logger.info(f"Received raw bytes, saved to: {temp_path}")
        return temp_path

    if content_type == "application/json":
        input_data = json.loads(request_body)

        if "file_path" in input_data:
            file_path = input_data["file_path"]

            if file_path.startswith("s3://"):
                logger.info(f"Downloading from S3: {file_path}")
                bucket, key = s3_uri_to_bucket_key(file_path)
                s3_client = boto3.client('s3')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                    temp_path = temp_file.name
                s3_client.download_file(bucket, key, temp_path)
                return temp_path

            elif os.path.exists(file_path):
                return file_path

            else:
                raise ValueError("Invalid file_path")

        elif "s3_uri" in input_data:
            s3_uri = input_data["s3_uri"]
            bucket, key = s3_uri_to_bucket_key(s3_uri)
            s3_client = boto3.client('s3')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_path = temp_file.name
            s3_client.download_file(bucket, key, temp_path)
            return temp_path

    elif content_type == "application/x-image":
        input_data = json.loads(request_body)
        image_data = base64.b64decode(input_data['file_path'])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        return temp_path

    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: Any, model_components: Dict[str, Any]) -> torch.Tensor:
    model = model_components["model"]
    device = model_components["device"]
    transforms = model_components["transforms"]

    image_tensor = transforms(input_data).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor[:, :, :, :, 0])

    return output

def output_fn(prediction: torch.Tensor, accept: str) -> str:
    probabilities = torch.softmax(prediction, dim=1)
    result = {
        "predictions": probabilities.cpu().numpy().tolist(),
        "predicted_class": probabilities.argmax(dim=1).cpu().numpy().tolist(),
        "confidence": probabilities.max(dim=1)[0].cpu().numpy().tolist()
    }
    if accept == "application/json;verbose":
        result["model_info"] = {"model_name": "DenseNet121", "num_classes": 8}
    return json.dumps(result)
