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
        self.logger.info(f"Submitting task {task_request['task_id']} to {self.submit_endpoint}")
        
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

def send_image_classification_request_squeezenet(counter: int, url: str) -> None:
    try:
        submitter = TaskSubmitter(url)
        image_path = "rhino.jpg"
        model_name = WASM_MODELS_FOLDER + "model_1/squeezenet1.1-7.onnx"
        labels_path = WASM_MODELS_FOLDER + "model_1/squeezenet1.1-7.txt"
        binary_name = "image_classification_squeezenet_onnx.wasm"
        model_folder_name = "model_1"
        task_request = encode_image_classification_task(image_path, model_name, labels_path,counter,binary_name,model_folder_name)
        result = submitter.submit_task(task_request)
    except Exception as e:
        print(f"Request {counter}: Error submitting task: {e}")
    
    print(f"Request {counter}: image_classification  task submitted successfully")

def send_image_classification_request_resnet(counter: int, url: str) -> None:
    try:
        submitter = TaskSubmitter(url)
        image_path = "rhino.jpg"
        model_name = WASM_MODELS_FOLDER + "model_3/resnet18.onnx"
        labels_path = WASM_MODELS_FOLDER + "model_3/squeezenet1.1-7.txt"
        binary_name = "image_classification_resnet_onnx.wasm"
        model_folder_name = "model_3"
        task_request = encode_image_classification_task(image_path, model_name, labels_path,counter,binary_name,model_folder_name)
        result = submitter.submit_task(task_request)
    except Exception as e:
        print(f"Request {counter}: Error submitting task: {e}")
    
    print(f"Request {counter}: image_classification  task submitted successfully")


def send_matmul_request(counter: int, url: str) -> None:
    """Send a matrix multiplication request.
    Args:
        counter: Current counter value
    """
    submitter = TaskSubmitter(url)
    matrix_a, matrix_b = generate_matmul_payload()
    
    task_request = encode_matrix_multiplication_request(matrix_a, matrix_b, counter)
    result = submitter.submit_task(task_request)
    
    print(f"Request {counter}: Matrix multiplication task submitted successfully")

def send_mat_transpose_request(counter: int, url: str) -> None:
    """Send a matrix multiplication request.
    Args:
        counter: Current counter value
    """
    submitter = TaskSubmitter(url)
    matrix_a, matrix_b = generate_matrix_transpose_payload()
    
    task_request = encode_matrix_transpose_request(matrix_a, counter)
    # print(task_request)
    result = submitter.submit_task(task_request)
    
    print(f"Request {counter}: Matrix transpose task submitted successfully")
    logger.info(f"Request {counter}: Task {task_request['task_id']} submitted")

def send_fibonacci_request(counter: int, url: str) -> None:
    """Send a matrix multiplication request.
    Args:
        counter: Current counter value
    """
    submitter = TaskSubmitter(url)
    n = 4
    task_request = encode_fibonacci_task(n, counter)
    result = submitter.submit_task(task_request)
    
    print(f"Request {counter}: Fibonacci task submitted successfully")

def send_fibonacci_optimized_request(counter: int, url: str) -> None:
    """Send a matrix multiplication request.
    Args:
        counter: Current counter value
    """
    submitter = TaskSubmitter(url)
    n = 55
    task_request = encode_fibonacci_optimized_request(n, counter)
    result = submitter.submit_task(task_request)
    
    print(f"Request {counter}: Fibonacci task submitted successfully")

def send_image_classification_request_squeezenet_batch(counter, url: str):
    """Send image classification request with batch of 5 identical images.
    Args:
        counter: Current counter value
    """
    try:
        submitter = TaskSubmitter(url)
        image_path = "rhino.jpg"
        model_name = WASM_MODELS_FOLDER+"model_1/squeezenet1.1-7.onnx"
        labels_path = WASM_MODELS_FOLDER+"model_1/squeezenet1.1-7.txt"
        binary_name = "image_classification_squeezenet_onnx_batch.wasm"
        model_folder_name = "model_1"
        cwasm_file = "image_classification_squeezenet_onnx_batch.cwasm",
        wat_file = "image_classification_squeezenet_onnx_batch.wat",
        
        # Read the same image 5 times
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Create array of 5 identical images
        image_payloads = []
        for _ in range(5):
            image_payloads.append(base64.b64encode(image_data).decode('utf-8'))
        
        # Convert payload to JSON string
        payload_dict = {
            "model_path": model_name,
            "labels_path": labels_path,
            "input": image_payloads
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
            "task_id": counter,
            "model_folder_name": model_folder_name,
            "cwasm_file":"image_classification_squeezenet_onnx_batch.cwasm",
            "wat_file": "image_classification_squeezenet_onnx_batch.wat",
        }
        
        result = submitter.submit_task(task_request)
    except Exception as e:
        print(f"Request {counter}: Error submitting batch task: {e}")
        logger.error(f"Request {counter}: Error submitting batch task: {e}")
    
    print(f"Request {counter}: image_classification batch task submitted successfully")
    logger.info(f"Request {counter}: Batch task {task_request['task_id']} submitted")

def send_image_classification_request_resnet_batch(counter, url: str):
    """Send image classification request with batch of 5 identical images.
    Args:
        counter: Current counter value
    """
    try:
        submitter = TaskSubmitter(url)
        image_path = "rhino.jpg"
        model_name = WASM_MODELS_FOLDER + "model_3/resnet18.onnx"
        labels_path = WASM_MODELS_FOLDER + "model_3/squeezenet1.1-7.txt"
        binary_name = "image_classification_resnet_onnx_batch.wasm"
        model_folder_name = "model_3"
        
        # Read the same image 5 times
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Create array of 5 identical images
        image_payloads = []
        for _ in range(5):
            image_payloads.append(base64.b64encode(image_data).decode('utf-8'))
        
        # Convert payload to JSON string
        payload_dict = {
            "model_path": model_name,
            "labels_path": labels_path,
            "input": image_payloads
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
            "task_id": counter,
            "model_folder_name": model_folder_name,
            "cwasm_file":"image_classification_resnet_onnx_batch.cwasm",
            "wat_file": "image_classification_resnet_onnx_batch.wat",
        }
        
        result = submitter.submit_task(task_request)
    except Exception as e:
        print(f"Request {counter}: Error submitting batch task: {e}")
        logger.error(f"Request {counter}: Error submitting batch task: {e}")
    
    print(f"Request {counter}: image_classification batch task submitted successfully")
    logger.info(f"Request {counter}: Batch task {task_request['task_id']} submitted")




def send_test_cases(url):
    counter = 0
    while True:
        send_image_classification_request_resnet_batch(counter, url)
        counter += 1
        send_image_classification_request_squeezenet_batch(counter, url)
        counter += 1
        send_matmul_request(counter, url)
        counter += 1
        send_image_classification_request_squeezenet_batch(counter, url)
        counter += 1
        send_mat_transpose_request(counter, url)
        counter += 1
        send_fibonacci_request(counter, url)
        counter += 1
        send_fibonacci_optimized_request(counter, url)
        counter += 1
        send_image_classification_request_squeezenet(counter)
        counter += 1
        send_image_classification_request_resnet(counter)
        counter += 1
        # # Uncomment to send prime number requests too:
        time.sleep(2)  # Sleep for 2 seconds
        break

