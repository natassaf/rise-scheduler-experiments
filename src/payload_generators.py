import base64
from typing import List
import random
from utils import SEED

random.seed(SEED)

def generate_matmul_payload(num_rows_1=None, num_rows_2=None, num_cols_2=None) -> tuple[List[List[float]], List[List[float]]]:
    
    num_rows_1 = random.randint(2, 1000)
    num_cols_1 = random.randint(2, 1000)
    num_rows_2 = num_rows_1
    num_cols_2 = random.randint(2, 1000)
    matrix_a = []
    for i in range(num_rows_1):
        row = []
        for j in range(num_cols_1):
            row.append(float(i + j))
        matrix_a.append(row)
    
    matrix_b = []
    for i in range(num_rows_2):
        row = []
        for j in range(num_cols_2):
            row.append(float(i * j))
        matrix_b.append(row)
    
    print(f"Created random matrices: A({len(matrix_a)}x{len(matrix_a[0])}), B({len(matrix_b)}x{len(matrix_b[0])})")
    return matrix_a, matrix_b


def generate_fibonacci_payload(n=None)-> int:
    n = random.randint(1, 50) if n is None else n
    return n


def generate_matrix_transpose_payload(num_rows=None, num_cols=None) -> List[List[float]]:
    num_rows = random.randint(2, 1000) if num_rows is None else num_rows
    num_cols = random.randint(2, 1000) if num_cols is None else num_cols
    matrix_a = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(float(i + j))
        matrix_a.append(row)
    

    
    print(f"Created random matrix: A({len(matrix_a)}x{len(matrix_a[0])})")
    return matrix_a

def generate_image_classification_payload(image_path=None) -> List[str]:
    image_path = f"../animal_data_sample/img{random.randint(1, 7)}.jpg" if image_path is None else image_path
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    image_payloads = []
    for _ in range(5):
        image_payloads.append(base64.b64encode(image_data).decode('utf-8'))
    
    return image_payloads