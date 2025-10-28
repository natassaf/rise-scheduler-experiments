

import base64
import gzip
import json
from typing import List, Dict, Any

from payload_generators import generate_image_classification_payload
from utils import WASM_MODELS_FOLDER

def encode_matrix_multiplication_request(matrix_a: List[List[float]], 
                                        matrix_b: List[List[float]], 
                                        task_id:int) -> Dict[str, Any]:

    # Convert payload to JSON string
    payload_dict = {
        "mat1": matrix_a,
        "mat2": matrix_b
    }
    payload_string = json.dumps(payload_dict)

    # Compress the entire JSON payload
    compressed_payload = gzip.compress(payload_string.encode('utf-8'))
    
    # Send as base64 string
    compressed_payload_b64 = base64.b64encode(compressed_payload).decode('utf-8')
    task_request = {
        "binary_name": "matrix_multiplication_component.wasm",
        "func_name": "run",
        "payload": compressed_payload_b64,
        "payload_compressed": True,
        "task_id": task_id,
        "model_folder_name": "",
        "cwasm_file":"matrix_multiplication_component.cwasm",
        "wat_file": "matrix_multiplication_component.wat",
    }
    
    print(f"Created matrix multiplication task: {task_id}")
    return task_request


def encode_fibonacci_request(n: float, task_id:int) -> Dict[str, Any]:
    """Create a fibanacci task request.
    
    Args:
        n: n-th fibonacci number
        
    Returns:
        Task request dictionary
    """
    
    # Convert payload to JSON string
    payload_dict = {
        "n": n,
    }
    payload_string = json.dumps(payload_dict)
    # No compression or base64 encoding needed
    task_request = {
        "binary_name": "fibonacci.wasm",
        "func_name": "run",
        "payload": payload_string,
        "payload_compressed": False,
        "task_id": task_id,
        "model_folder_name": "",
        "cwasm_file":"fibonacci.cwasm",
        "wat_file": "fibonacci.wat",
    }
    
    print(f"Created fibonacci task: {task_id}")
    return task_request


def encode_fibonacci_optimized_request(n: float, task_id:int) -> Dict[str, Any]:
    # Convert payload to JSON string
    payload_dict = {
        "n": n,
    }
    payload_string = json.dumps(payload_dict)
    # No compression or base64 encoding needed
    task_request = {
        "binary_name": "fibonacci_optimized.wasm",
        "func_name": "run",
        "payload": payload_string,
        "payload_compressed": False,
        "task_id": task_id,
        "model_folder_name": "",
        "cwasm_file":"fibonacci_optimized.cwasm",
        "wat_file": "fibonacci_optimized.wat",
    }
    
    print(f"Created matrix multiplication task: {task_id}")
    return task_request


def encode_matrix_transpose_request(matrix: List[List[float]], task_id:int) -> Dict[str, Any]:
    """Create a matrix transpose task request.
    
    Args:
        matrix: (2D list of floats)
        
    Returns:
        Task request dictionary
    """
    
    # Convert payload to JSON string
    payload_dict = {
        "matrix": matrix,
    }
    payload_string = json.dumps(payload_dict)
    # Compress the entire JSON payload
    compressed_payload = gzip.compress(payload_string.encode('utf-8'))
    # Send as base64 string
    compressed_payload_b64 = base64.b64encode(compressed_payload).decode('utf-8')
    task_request = {
        "binary_name": "matrix_transpose.wasm",
        "func_name": "run",
        "payload": compressed_payload_b64,
        "payload_compressed": True,
        "task_id": task_id,
        "model_folder_name": "",
        "cwasm_file":"matrix_transpose.cwasm",
        "wat_file": "matrix_transpose.wat",
    }
    
    print(f"Created matrix multiplication task: {task_id}")
    return task_request




def encode_image_classification_task(image_path: str, 
                                    model_path: str,
                                    labels_path: str,
                                    task_id:int,
                                    binary_name: str,
                                    model_folder_name: str) -> Dict[str, Any]:
    """Create an image classification task request.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the classification model to use
        
    Returns:
        Task request dictionary
    """
    with open(image_path, "rb") as image_file:
        image_payload = base64.b64encode(image_file.read()).decode('utf-8')

    # Convert payload to JSON string
    payload_dict = {
        "model_path": model_path,
        "labels_path": labels_path,
        "input": image_payload
    }
    payload_string = json.dumps(payload_dict)
    
    # Compress the entire JSON payload
    compressed_payload = gzip.compress(payload_string.encode('utf-8'))
    # Send as base64 string
    compressed_payload_b64 = base64.b64encode(compressed_payload).decode('utf-8')
    
    task_request = {
        "binary_name": binary_name,
        "func_name": "run",
        "payload": compressed_payload_b64,
        "payload_compressed": True,
        "task_id": task_id,
        "model_folder_name": model_folder_name
    }
    
    print(f"Created image classification task: {task_id}")
    return task_request

def encode_image_classification_request_squeezenet_batch(image_path: str, task_id: int) -> Dict[str, Any]:
    """Send image classification request with batch of 5 identical images.
    Args:
        counter: Current counter value
    """
    model_name = WASM_MODELS_FOLDER+"model_1/squeezenet1.1-7.onnx"
    labels_path = WASM_MODELS_FOLDER+"model_1/squeezenet1.1-7.txt"
    binary_name = "image_classification_squeezenet_onnx_batch.wasm"
    model_folder_name = "model_1"
    cwasm_file = "image_classification_squeezenet_onnx_batch.cwasm"
    wat_file = "image_classification_squeezenet_onnx_batch.wat"
    
    payload = generate_image_classification_payload()

    # Convert payload to JSON string
    payload_dict = {
        "model_path": model_name,
        "labels_path": labels_path,
        "input": payload
    }
    payload_string = json.dumps(payload_dict)
    
    # Compress the entire JSON payload
    compressed_payload = gzip.compress(payload_string.encode('utf-8'))
    # Send as base64 string
    compressed_payload_b64 = base64.b64encode(compressed_payload).decode('utf-8')
    
    task_request = {
        "binary_name": binary_name,
        "func_name": "run",
        "payload": compressed_payload_b64,
        "payload_compressed": True,
        "task_id": task_id,
        "model_folder_name": model_folder_name,
        "cwasm_file":cwasm_file,
        "wat_file": wat_file,
    }
    return task_request

def encode_image_classification_request_resnet_batch(image_path: str, task_id: int):
    model_name = WASM_MODELS_FOLDER + "model_3/resnet18.onnx"
    labels_path = WASM_MODELS_FOLDER + "model_3/squeezenet1.1-7.txt"
    binary_name = "image_classification_resnet_onnx_batch.wasm"
    model_folder_name = "model_3"
    
    payload = generate_image_classification_payload()

    # Convert payload to JSON string
    payload_dict = {
        "model_path": model_name,
        "labels_path": labels_path,
        "input": payload
    }
    payload_string = json.dumps(payload_dict)
    
    # Compress the entire JSON payload
    compressed_payload = gzip.compress(payload_string.encode('utf-8'))
    # Send as base64 string
    compressed_payload_b64 = base64.b64encode(compressed_payload).decode('utf-8')
    
    task_request = {
        "binary_name": binary_name,
        "func_name": "run",
        "payload": compressed_payload_b64,
        "payload_compressed": True,
        "task_id": task_id,
        "model_folder_name": model_folder_name,
        "cwasm_file":"image_classification_resnet_onnx_batch.cwasm",
        "wat_file": "image_classification_resnet_onnx_batch.wat",
    }
    
    return task_request

