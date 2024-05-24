import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--f", help="yaml file path")
parser.add_argument("--d", default='cuda:0', help="device")
parser.add_argument("--e", help="experiment directory")
args = parser.parse_args()
yaml_filepath = args.f
device = args.d
experiment = args.e

if "maf" in yaml_filepath:
    os.system("python main_maf.py --f {} --d {} --e {}".format(yaml_filepath, device, experiment))
    
elif "wgan" in yaml_filepath:
    os.system("python main_wgan.py --f {} --d {} --e {}".format(yaml_filepath, device, experiment))
    
elif "vae" in yaml_filepath:
    os.system("python main_vae.py --f {} --d {} --e {}".format(yaml_filepath, device, experiment))