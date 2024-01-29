import tvm
from tvm import rpc
import tvm.relay as relay
from tvm.relay import transform
from tvm.relay.dataflow_pattern import rewrite

import numpy as np

import json
import traceback

import os
from os import path
from os import listdir
from os.path import isfile, isdir, join

from mutators.generators import model as model_generator
from executors import mutations
from evaluators.evalgenerator import EvaluationGenerator
from loaders.model_loader import ModelLoader

import numpy as np
from numpy.linalg import norm

from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
import pandas as pd

from matplotlib.lines import Line2D

script_dir = os.path.dirname(os.path.realpath(__file__))

def props(cls):
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def load_config():
    with open('./config.json') as f:
        return json.load(f)

def quantize(mod, params):
    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
        mod = relay.quantize.quantize(mod, params)
    return mod

config = load_config()

# Generic Configuration
device_name = config["devices"]["selected"]
build = config["devices"][device_name]
build["device_name"] = device_name
build["id"] = config["devices"]["id"]
image_count_threshold = config["image_count_threshold"] if "image_count_threshold" in config else 0

# ----- Setup Target Configuration -----
target = tvm.target.Target(build["target"], host=build["host"])

mutations_info = config["mutations"]

datasets_info = config["datasets"]
default_dataset_info = datasets_info[config["selected_dataset"]]

images_path = script_dir + "/" + default_dataset_info["dataset_path_relative"]
image_names  = [f for f in listdir(images_path) if isfile(join(images_path, f))]

evaluation_base_folder = "/mutations/ts_full/"
evaluation_path = evaluation_base_folder + "/evaluate_mutation.txt"
evaluation_single_device_path = evaluation_base_folder + "/evaluate_single_device_mutation.txt"

mutation_model_evaluations = {}
model_names = []

print ("Preprocessing is " + ("enabled." if config["preprocessing_enabled"] else "disabled."))

model_loader = ModelLoader()
mutations_executor = None

# ***** Prepare Device Configuration *****
host_type = build["host_type"]
device_id = build["id"]
if(host_type == "local_no_rpc"):
    remote = None
elif (host_type == "local"):
    print ("Preparing using Local RPC connection. - Device ID: " + str(device_id))
    remote = rpc.LocalSession()
else:
    address = build["address"]
    port = build["port"]
    print ("Preparing on : " + address + ":" + str(port) + " - Device ID: " + str(device_id))
    remote = rpc.connect(address, port)


for loop_count in range(config["runs_no"]):
    print("Run " + str(loop_count + 1) + " out of " + str(config["runs_no"]) + ".")
    for model_info in config["models"]:

        if("skip_analysis" in model_info and model_info["skip_analysis"]):
            print("Skipping analysis on " + model_info["alias"])
            continue

        print("Processing model " + model_info["alias"])

        input_name = model_info["input_name"]
        output_name = model_info["output_name"] if "output_name" in model_info else None
        paths_info = model_info["paths"]
        models_path = script_dir + "/" + paths_info["models_out_relative"]
        generated_path = script_dir + "/generated/"

        input_layer_shape = tuple(model_info["layers"]["input"])
        x = np.zeros(shape=input_layer_shape)
        shape_dict = {input_name: x.shape}

        out_path = script_dir + paths_info["exec_out_relative"]
        error_base_folder = join(script_dir, "error_log", model_info["name"])

        models_data = {
            "name": model_info["name"],
            "url": model_info["url"],
            "alias": model_info["alias"],
            "type": model_info["type"] if "type" in model_info else "local",
            "model": model_info["model"] if "model" in model_info else "",
            "model_name": model_info["name"],
            "raw_model_name": model_info["name"],
            "input_model_folder": models_path,
            "output_model_folder": models_path,
            "image_dimension": tuple(model_info["layers"]["image_dimension"]),
            "input": tuple(model_info["layers"]["input"]),
            "input_name": input_name,
            "output": tuple(model_info["layers"]["output"]),
            "output_name": output_name,
            "library": model_info["library"] if "library" in model_info else None,
            "preprocessing_enabled": config["preprocessing_enabled"],
            "debug_enabled": config["debug_enabled"],
            "dtype": model_info["dtype"] if "dtype" in model_info else "float32"
        }

        # ----- Model Building & Mutations Generation -----
        if(loop_count == 0 and ("build" in model_info and model_info["build"])):
            print("Building: " + model_info["name"])
            os.makedirs(models_path, exist_ok=True)

            model_type = model_info["type"] if "type" in model_info else "local"

            mod, params = model_loader.load_model(models_data, model_type=model_type)

            if("quantize" in config and config["quantize"]):
                print("Quantization enabled.")
                mod = quantize(mod, params)

            model_generator.generate_original_model(mod, target, params, paths_info["models_out_relative"], \
                opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], \
                    quantize=config["quantize"], opt_alias=config["opt_alias"])
            
            if ("enable_mutations" in config and config["enable_mutations"]):

                # Apply mutations ONLY in specific positions.
                if ("positions" in mutations_info):
                    info = mutations_info["positions"][model_info["name"]]
                    relay_positions = info["relay"]
                    for relay_position in relay_positions:
                        for mutation in mutations_info["relay"]:
                            mutation["start"] = relay_position
                            mutation["end"] = relay_position
                        model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], \
                            opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], \
                                quantize=config["quantize"], opt_alias=config["opt_alias"])
                        
                    tir_positions = info["tir"]
                    for tir_position in tir_positions:
                        for mutation in mutations_info["tir"]:
                            mutation["start"] = tir_position
                            mutation["end"] = tir_position
                        model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], \
                            opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], \
                                quantize=config["quantize"], opt_alias=config["opt_alias"])
            
                else:
                    model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], \
                        opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], \
                            quantize=config["quantize"], opt_alias=config["opt_alias"])
                    model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], \
                        opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], \
                            quantize=config["quantize"], opt_alias=config["opt_alias"])

        mutations_names = []
        if (path.exists(models_path)):
            mutations_names = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith(".tar") and ("_ignore_" not in f)]

        mutations_names.sort(key=lambda x: "original" in x, reverse=True)
        models_data["mutations_names"] = mutations_names

        print("Mutations generated:")
        print(mutations_names)
        
        folders_to_execute = []
        folders_to_execute.append(images_path)

        images_data = {
            "input_images_folders": folders_to_execute,
            "output_images_base_folder": out_path
        }

        build["error_base_folder"] = error_base_folder

        # ----- Execution & Inference -----
        try:
            # Direct NN model Execution from ONNX.
            if("execute" in model_info and model_info["execute"]):

                print("Executing: " + model_info["name"])

                # Defined here to allow multiple executions.
                if mutations_executor is None:
                    mutations_executor = mutations.MutationsExecutor(models_data, images_data, build, image_count_threshold=image_count_threshold)
                mutations_executor.execute(remote)

        except Exception as e:
            print(traceback.print_exc())

        # ***** Analyze & Evaluate Executions *****
        if ("evaluate" in model_info and model_info["evaluate"]):
            model_base = script_dir + "/" + paths_info["evaluation_out_relative"]
            mutations_names = [f for f in listdir(model_base) if isfile(join(model_base, f)) and f.endswith(".tar") and ("_ignore_" not in f)]
            generated_models_prettified = [model.replace(".tar", "").replace(".", "_") for model in mutations_names]

            # Set a threshold for evaluator, so that under this, the batch inference is considered as "crashed".
            # This is because, for big datasets, the inference generator smartly detects similar occurences of outputs
            # and stops the inference if this threshold is passed.
            eg_mts = EvaluationGenerator(image_count_threshold=image_count_threshold)
            
            print("Evaluating: " + model_info["name"])

            device_index = 0
            device_folders = [d for d in listdir(model_base) if isdir(join(model_base, d))]

            for device_folder in device_folders:

                # Compare across ground truth.
                eg_mts.get_same_folder_comparison(join(model_base, device_folder), "model_original")
                # Compare across devices.
                eg_mts.generate_devices_comparison(model_base, replace_evaluated_suffix=True)                
                eg_mts.distinguish_device_discrepancies(model_base)
