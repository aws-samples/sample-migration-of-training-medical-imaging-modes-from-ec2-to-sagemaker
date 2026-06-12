"""Inference script for medical image classification models."""

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
    """Parse S3 URI to extract bucket and key."""
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load the medical image classification model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        logger.info(f"Files in model_dir: {os.listdir(model_dir)}")
        
        model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=8)
        
        model_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(model_path):
            logger.info(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Model weights loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
        
        model.to(device).eval()
        
        transforms = Compose([
            LoadImage(),
            EnsureChannelFirst(),
            Resize(spatial_size=(256, 256, 1)),
            ScaleIntensity()
        ])
        
        return {"model": model, "device": device, "transforms": transforms}
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body: Union[str, bytes], content_type: str) -> Any:
    """Parse input data for inference."""
    if content_type == "application/json":
        input_data = json.loads(request_body)
        
        if "file_path" in input_data:
            file_path = input_data["file_path"]
            
            if file_path.startswith("s3://"):
                try:
                    logger.info(f"Downloading from S3: {file_path}")
                    bucket, key = s3_uri_to_bucket_key(file_path)
                    s3_client = boto3.client('s3')
                    
                    file_extension = '.dcm'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_path = temp_file.name
                    
                    s3_client.download_file(bucket, key, temp_path)
                    logger.info(f"Downloaded to: {temp_path}")
                    return temp_path
                    
                except Exception as e:
                    logger.error(f"S3 download error: {str(e)}")
                    raise ValueError(f"S3 download failed: {str(e)}")
            
            elif os.path.exists(file_path):
                return file_path
            
            else:
                raise ValueError("Invalid file_path")
        
        elif "s3_uri" in input_data:
            s3_uri = input_data["s3_uri"]
            logger.info(f"Downloading from S3: {s3_uri}")
            bucket, key = s3_uri_to_bucket_key(s3_uri)
            s3_client = boto3.client('s3')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_path = temp_file.name
            
            s3_client.download_file(bucket, key, temp_path)
            return temp_path
    
    elif content_type == "application/x-image":
        try:
            input_data = json.loads(request_body)
            logger.info("Decoding base64 image")
            image_data = base64.b64decode(input_data['file_path'])
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_file:
                temp_path = temp_file.name
                temp_file.write(image_data)
                temp_file.flush()
            
            logger.info(f"Base64 decoded to: {temp_path}")
            return temp_path
        
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            raise ValueError("Invalid base64 data")
    
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: Any, model_components: Dict[str, Any]) -> torch.Tensor:
    """Run inference on medical image."""
    try:
        model = model_components["model"]
        device = model_components["device"]
        transforms = model_components["transforms"]
        
        logger.info(f"Processing image: {input_data}")
        image_tensor = transforms(input_data).unsqueeze(0).to(device)
        
        logger.info("Running inference")
        with torch.no_grad():
            output = model(image_tensor[:, :, :, :, 0])
        
        logger.info("Inference completed")
        return output
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise

def output_fn(prediction: torch.Tensor, accept: str) -> Union[str, bytes]:
    """Format prediction output."""
    if accept == "application/json":
        probabilities = torch.softmax(prediction, dim=1)
        result = {
            "predictions": probabilities.cpu().numpy().tolist(),
            "predicted_class": probabilities.argmax(dim=1).cpu().numpy().tolist(),
            "confidence": probabilities.max(dim=1)[0].cpu().numpy().tolist()
        }
        return json.dumps(result)
    
    elif accept == "application/json;verbose":
        probabilities = torch.softmax(prediction, dim=1)
        result = {
            "predictions": probabilities.cpu().numpy().tolist(),
            "predicted_class": probabilities.argmax(dim=1).cpu().numpy().tolist(),
            "confidence": probabilities.max(dim=1)[0].cpu().numpy().tolist(),
            "model_info": {
                "model_name": "DenseNet121",
                "num_classes": 8
            }
        }
        return json.dumps(result, indent=2)
    


def test_local_file():
    """Test function for local file inference."""
    sample_path = './samples/sample_image.dcm'
    input_data = {"file_path": sample_path}
    input_dump = json.dumps(input_data)
    
    try:
        input_data = input_fn(input_dump, "application/json")
        logger.info(f"Input processed: {input_data}")
        
        model_dir = './model'
        model_components = model_fn(model_dir=model_dir)
        result = predict_fn(input_data, model_components)
        output = output_fn(result, "application/json")
        
        logger.info(f"Prediction result: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in local test: {str(e)}", exc_info=True)
        return {"error": str(e)}

def test_s3_file():
    """Test function for S3 file inference."""
    # sample_path = "s3://public-datasets-imaging-us-east-1/vindr-spinexr-subset/train/Normal/sample.dcm"
    sample_path = "s3://public-datasets-imaging/vindr-spine-raw/files/vindr-spinexr/1.0.0/test_images/000b3dad09378f680c845f8d7827d6ad.dicom"
    input_data = {"file_path": sample_path}
    input_dump = json.dumps(input_data)
    
    try:
        input_data = input_fn(input_dump, "application/json")
        logger.info(f"Input processed: {input_data}")
        
        model_dir = './model'
        model_components = model_fn(model_dir=model_dir)
        result = predict_fn(input_data, model_components)
        output = output_fn(result, "application/json;verbose")
        
        logger.info(f"Prediction result: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in S3 test: {str(e)}")
        return {"error": str(e)}

def test_base64_file():
    """Test function for base64 encoded file inference."""
    try:
        with open('./samples/sample_image.dcm', 'rb') as f:
            encoded_data = base64.b64encode(f.read()).decode('utf-8')
        
        input_data = {"file_path": encoded_data}
        input_dump = json.dumps(input_data)
        
        input_data = input_fn(input_dump, "application/x-image")
        logger.info(f"Base64 input processed: {input_data}")
        
        model_dir = './model'
        model_components = model_fn(model_dir=model_dir)
        result = predict_fn(input_data, model_components)
        output = output_fn(result, "application/json")
        
        logger.info(f"Base64 prediction result: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in base64 test: {str(e)}")
        return {"error": str(e)}

def test_model_loading():
    """Test model loading functionality."""
    try:
        model_dir = './model'
        model_components = model_fn(model_dir=model_dir)
        
        assert "model" in model_components
        assert "device" in model_components
        assert "transforms" in model_components
        
        logger.info("Model loading test passed")
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return {"status": "error", "message": str(e)}

def run_all_tests():
    """Run all test functions."""
    logger.info("Running all inference tests")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Local File", test_local_file),
        ("S3 File", test_s3_file),
        ("Base64 File", test_base64_file)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        results[test_name] = test_func()
    
    return results

def main():
    """Main function for testing."""
    logger.info("Testing medical image classification inference")
    results = test_s3_file()
    
    # logger.info("Test Results Summary:")
    # for test_name, result in results.items():
    #     if isinstance(result, str):
    #         result = json.loads(result)
    #     status = "PASS" if "error" not in result else "FAIL"
    #     logger.info(f"  {test_name}: {status}")
    
    return results

if __name__ == '__main__':
    main()
