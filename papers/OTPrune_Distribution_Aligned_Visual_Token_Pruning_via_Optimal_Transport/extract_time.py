import re
import argparse

def extract_after_generation_memory(line):
    match = re.search(r'after generation memory: (\d+)', line)
    if match:
        return int(match.group(1))
    return None

def extract_generation_latency_time(line):
    match = re.search(r'Generation latency time is: ([\d.]+)', line)
    if match:
        return float(match.group(1))
    return None

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    feature_size_array = []
    after_generation_memory_array = []
    generation_latency_array = []
    i = 0
    while i < len(lines):

        after_generation_memory = extract_after_generation_memory(lines[i])
        if after_generation_memory:
            after_generation_memory_array.append(after_generation_memory)

        generation_latency_time = extract_generation_latency_time(lines[i])
        if generation_latency_time:
            past_latency_time = generation_latency_time
            generation_latency_array.append(generation_latency_time)       
        i += 1
    
    return after_generation_memory_array,generation_latency_array

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="input path arg")
    parser.add_argument('--path', type=str, help='path to the log', default = './log_eval.log')
    args = parser.parse_args()

    memoery_array, latency_array = process_file(args.path)
    print(f"Average max memory: {float(sum(memoery_array))/len(memoery_array)} Bytes")
    print(f"Average latency: {float(sum(latency_array))/len(latency_array)} mSces" )