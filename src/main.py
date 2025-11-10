
import json
import logging
import time
import os
from typing import Dict, Any
import base64
import requests
import gzip

from test_case_generators import generate_all_test_cases
from utils import WASM_MODELS_FOLDER, load_json_file, parse_file_names



class TaskSubmitter:
    """Class for submitting different types of tasks to the server."""
    
    def __init__(self, url: str):
        """Initialize the task submitter.
        
        Args:
            base_url: Base URL of the task submission server
        """
        self.base_url = url
        self.submit_endpoint = f"{url}/submit_task"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    

    def submit_task(self, task_request: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = requests.post(
            self.submit_endpoint,
            json=task_request,
            headers=headers,
            timeout=30
        )
        print("respomse",response.text)
        
        print(f"Task {task_request['task_id']} submitted successfully")
        return response


def parse_task_id(json_file: str) -> str:
    return json_file.split("/")[-1].split(".")[0]

def send_matmul_request(url: str) -> None:
    """Send matrix multiplication requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/matmul"
    
    # Get all files in the matmul directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} matmul test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)

            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            time.sleep(0.2)
            print(f"Request {counter}_{i}: Matrix multiplication task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")

def send_mat_transpose_request(counter: int, url: str) -> None:
    """Send matrix transpose requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/matrix_transpose"
    
    # Get all files in the matrix_transpose directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} matrix transpose test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)
            
            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            
            print(f"Request {counter}_{i}: Matrix transpose task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")

def send_fibonacci_request(counter: int, url: str) -> None:
    """Send fibonacci requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/fibonacci"
    
    # Get all files in the fibonacci directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} fibonacci test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)
            
            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            time.sleep(3)
            print(f"Request {counter}_{i}: Fibonacci task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")

def send_fibonacci_optimized_request(counter: int, url: str) -> None:
    """Send fibonacci optimized requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/fib_opt"
    
    # Get all files in the fib_opt directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} fibonacci optimized test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)
            
            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            print(f"Request {counter}_{i}: Fibonacci optimized task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")

def send_image_classification_request_squeezenet_batch(counter, url: str):
    """Send image classification squeezenet batch requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/image_classification_squeezenet"
    
    # Get all files in the image_classification_squeezenet directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} image classification squeezenet test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)
            
            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            time.sleep(0.2)
            print(f"Request {counter}_{i}: Image classification squeezenet task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")

def send_image_classification_request_resnet_batch(counter, url: str):
    """Send image classification resnet batch requests from test cases.
    Args:
        counter: Current counter value
        url: Server URL
    """
    submitter = TaskSubmitter(url)
    test_cases_dir = "test_data/image_classification_resnet"
    
    # Get all files in the image_classification_resnet directory using parse_file_names
    json_files = parse_file_names(test_cases_dir)
    
    if not json_files:
        print(f"No test case files found in {test_cases_dir}")
        return
    
    print(f"Found {len(json_files)} image classification resnet test cases")
    
    for i, json_file in enumerate(json_files):
        try:
            # Load the task request from JSON file
            task_request = load_json_file(json_file)
            
            # Get task_id
            task_id = parse_task_id(json_file)
            task_request["task_id"] = f"{task_id}"
            
            # Submit the task
            result = submitter.submit_task(task_request)
            time.sleep(0.2)
            print(f"Request {counter}_{i}: Image classification resnet task from {os.path.basename(json_file)} submitted successfully")
            
        except Exception as e:
            print(f"Request {counter}_{i}: Error submitting task from {json_file}: {e}")




def send_test_cases(url):
    counter = 0
    while True:
        send_fibonacci_optimized_request(counter, url)

        # send_image_classification_request_resnet_batch(counter, url)
        # counter += 1
        # # print(counter)
        # send_image_classification_request_squeezenet_batch(counter, url)
        # counter += 1
        send_matmul_request(url)
        # send_mat_transpose_request(counter, url)
        # counter += 1
        # send_fibonacci_request(counter, url)
        # time.sleep(2) 
        break
    print(counter)
    

if __name__ == "__main__":
    url = "http://127.0.0.1:8080"
    # url="http://192.168.8.110:8082"
    send_test_cases(url)
    # time.sleep(1000)
    # generate_all_test_cases(25)
