# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:36:25 2024

@author: MaxGr
"""

import os
import pandas as pd
import re

import numpy as np


import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_patch", type=str, default="output0.05", help="path to model log")
parser.add_argument("--output_path", type=str, default="output_csv", help="path to output all csv")

args = parser.parse_args()


save_dir = os.path.join(args.output_path,args.save_patch)
os.makedirs(save_dir,exist_ok=True)
# Example usage
root_directory = args.save_patch  # Replace with the actual root directory
print(root_directory)
# log_file = 'output0\\base2new\\test_new\\ucf101\\shots_16\\MaPLe\\vit_b16_c2_ep5_batch4_2ctx\\seed2/log.txt'
def extract_log_param(log_file):
    # Patterns to extract the desired information
    epoch_pattern = r"model\.pth\.tar-(\d+)"  # Matches epoch number
    accuracy_pattern = r"accuracy: (\d+\.\d+)%"
    error_pattern = r"error: (\d+\.\d+)%"
    macro_f1_pattern = r"macro_f1: (\d+\.\d+)%"
    # Storage for results
    results = []
    # Read the log file line by line
    with open(log_file, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                # Look ahead for accuracy, error, and macro_f1
                accuracy_match = re.search(accuracy_pattern, lines[i + 5])
                error_match = re.search(error_pattern, lines[i + 6])
                macro_f1_match = re.search(macro_f1_pattern, lines[i + 7]) 
                
                accuracy = float(accuracy_match.group(1)) if accuracy_match else None
                error = float(error_match.group(1)) if error_match else None
                macro_f1 = float(macro_f1_match.group(1)) if macro_f1_match else None
                
                if accuracy:
                    results.append({
                        "epoch": epoch,
                        "accuracy": accuracy,
                        "error": error,
                        "macro_f1": macro_f1
                    })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    print(f"Results retrived from {log_file}")
    return results, df_results

def find_batch(text):
    pattern = r"batch(\d+)"
    match = re.search(pattern, text)
    if match:
        batch_size = int(match.group(1))
    return batch_size

def harmonic_mean(data):
    if not data or any(x == 0 for x in data):
        raise ValueError("Data must be non-empty and contain no zeros")
    n = len(data)
    return n / sum(1 / x for x in data)

# List to store log file information
log_files = []
# log_copys = []

# Traverse directories to find all log files
for dirpath, dirnames, filenames in os.walk(root_directory):
    # print(dirpath)
    # copy = 1 if dirpath[-6:] == '-Copy1' else 0
    copy = False
    for file in filenames:
        if file.endswith(".txt") and "log" in file:

            if file == 'log-checkpoint.txt':
                continue
                # copy = 'checkpoint'
            
            relative_path = os.path.relpath(dirpath, root_directory)
            components = relative_path.split(os.sep)
            log_file_path = os.path.join(dirpath, file)
            
            # Find batch 
            cfg = components[5]
            batch_size = find_batch(cfg)
            
            if dirnames == ['.ipynb_checkpoints']:
                print(cfg)
                copy = cfg[-6:]

            file_data = {
                "path": log_file_path,
                "file": file,
                "task_name": components[0],
                "novel": components[1],
                "dataset": components[2],
                "shots": components[3],
                "trainer": components[4],
                "cfg": components[5],
                "seed": components[6],
                "copy": copy
                # "details": "/".join(components[4:]),
            }
            
            # Get epoch results from log
            results, df_results = extract_log_param(log_file_path)
            
            acc_list = []
            for result in results:
                result["batch_size"] = batch_size
                
                
                acc = result["accuracy"]/100
                acc_list.append(acc)
                # h_mean = harmonic_mean(acc_list)
                # result["harmonic_mean"] = h_mean
                
                # Merge all info
                # info_total = file_data | result
                
                info_total = {**file_data, **result}
                
                # if copy:
                #     log_copys.append(info_total)
                # else:
                log_files.append(info_total)

# Convert to DataFrame
df_logs = pd.DataFrame(log_files)
# df_copy = pd.DataFrame(log_copys)

# with pd.ExcelWriter(F"{root_directory}_log_files_summary.xlsx", engine='xlsxwriter') as writer:
for config_i in df_logs['cfg'].unique():
    print(config_i)
    df_i = df_logs[df_logs['cfg']==config_i]
    sheet_name = config_i[:31]
    # df_i.to_excel(F"{config_i}_log_files_summary.xlsx", sheet_name=f'{sheet_name}', index=False)
    df_i.to_excel(os.path.join(save_dir,F"{config_i}.xlsx"), index=False)#, sheet_name=f'{sheet_name}'

    
print(f"Log files information saved")
