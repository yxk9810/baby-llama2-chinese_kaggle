import json
import glob
import torch
import numpy as np
import pandas as pd
import os


def check_is_processed(target_name, data_path):
    data_path_list = os.listdir(data_path)
    for data_name in data_path_list:
        if target_name.split('/')[-1] in data_name:
            return True

    return False

def getAllFiles(data_path):
    all_data_path_list = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            all_data_path_list.append(file_path)
    return all_data_path_list
