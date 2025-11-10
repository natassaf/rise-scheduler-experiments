from encoders_decoders import  encode_fibonacci_optimized_request, encode_fibonacci_request, encode_image_classification_request_resnet_batch, encode_image_classification_request_squeezenet_batch, encode_matrix_multiplication_request, encode_matrix_transpose_request
from payload_generators import generate_fibonacci_payload, generate_image_classification_payload, generate_matmul_payload
from utils import save_json_file


def generate_and_save_matmul_test_cases(num_cases_per_task: int):
    counter = 0
    name = "matmul"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        matrix_a, matrix_b = generate_matmul_payload()
        task_request = encode_matrix_multiplication_request(matrix_a, matrix_b, uuid)
        save_json_file(f"test_data/matmul/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)

def generate_and_save_fibonacci_test_cases(num_cases_per_task: int):
    counter = 0
    name = "fib"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        n = generate_fibonacci_payload()
        task_request = encode_fibonacci_request(n, uuid)
        save_json_file(f"test_data/fibonacci/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)

def generate_and_save_fibonacci_optimized_test_cases(num_cases_per_task: int):
    counter = 0
    name = "fib_opt"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        n = generate_fibonacci_payload()
        task_request = encode_fibonacci_optimized_request(n, uuid)
        save_json_file(f"test_data/{name}/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)

def generate_matrix_transpose_test_cases(num_cases_per_task: int):
    counter = 0
    name = "matrix_transpose"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        matrix_a= generate_matmul_payload()
        task_request = encode_matrix_transpose_request(matrix_a, uuid)
        save_json_file(f"test_data/{name}/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)


def generate_image_classification_squeezenet_test_cases(num_cases_per_task: int):
    counter = 0
    name = "image_classification_squeezenet"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        payload = generate_image_classification_payload()
        task_request = encode_image_classification_request_squeezenet_batch(payload, uuid)
        save_json_file(f"test_data/{name}/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)

def generate_image_classification_resnet_test_cases(num_cases_per_task: int):
    counter = 0
    name = "image_classification_resnet"
    uuid = name + str(counter)
    for j in range(num_cases_per_task):
        payload = generate_image_classification_payload()
        task_request = encode_image_classification_request_resnet_batch(payload, uuid)
        save_json_file(f"test_data/{name}/test_case_{uuid}.json", task_request)
        counter += 1
        uuid = name + str(counter)

def generate_all_test_cases(num_cases_per_task: int):
    generate_and_save_matmul_test_cases(num_cases_per_task)
    # generate_and_save_fibonacci_test_cases(num_cases_per_task)
    # generate_and_save_fibonacci_optimized_test_cases(num_cases_per_task)
    # generate_matrix_transpose_test_cases(num_cases_per_task)
    # generate_image_classification_squeezenet_test_cases(num_cases_per_task)
    # generate_image_classification_resnet_test_cases(num_cases_per_task)