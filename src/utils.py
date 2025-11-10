import json
from typing import Any, Dict
import os
import glob

TRAINING_SEED = 42
TESTING_SEED=100
SEED = TRAINING_SEED
# WASM_MODELS_FOLDER = "/home/pi/memory-estimator/models/" 
WASM_MODELS_FOLDER = "/Users/athanasiapharmake/workspace/rise-thesis/models" 
import os
import json

def save_json_file(file_path, data):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def load_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        return json.load(file)


def parse_file_names(folder_directory: str) -> list[str]:
    """Parse file names from a directory and return a list of file paths.
    
    Args:
        folder_directory: Path to the directory to parse
        
    Returns:
        List of file paths found in the directory
    """

    
    # Get all files in the directory
    file_pattern = os.path.join(folder_directory, "*")
    files = glob.glob(file_pattern)
    
    # Filter out directories and return only files
    file_names = [f for f in files if os.path.isfile(f)]
    
    return file_names
