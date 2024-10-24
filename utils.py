import os
import json

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return json.load(f)

def dump_json(file_path, data):
    make_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def dump_file(file_path, data):
    make_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        f.write(data)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)