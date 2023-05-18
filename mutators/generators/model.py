import tvm
from tvm import te
import tvm.relay as relay
import tvm.runtime as runtime

import time
import os

import numpy as np
import json
import onnx

from mutators.core.relay import mutate as mutate_relay
from mutators.core.tir import mutate as mutate_tir
from mutators.core.tir import save_original_pass

script_dir = os.path.dirname(os.path.realpath(__file__))




def get_code_dir_path(model_name, out_path):
    return script_dir + "/../../" + out_path + "/" + model_name + "_code"
    

def generate_original_model(model, target, params, out_path_relative, file_name=None, opt_level=3, required_pass=None, disabled_pass=None, quantize=False, opt_alias="", lib_params=None):
    print("Generating original model.")
    print("Folder: " + out_path_relative)
    

    model_name = file_name if file_name is not None else "model_original"
    
    model_name = model_name if not quantize else model_name + "_quant" 
    model_name = model_name + "_opt" + str(opt_level)
    model_name = model_name + opt_alias
    save_original_pass.mutation_data = {
        "model_name": model_name,
        "out_path_relative": out_path_relative
    }

    os.makedirs(get_code_dir_path(model_name, out_path_relative), exist_ok=True)

    output_tir_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_tir.txt",'w').close()
    os.makedirs(get_code_dir_path(model_name, out_path_relative), exist_ok=True)
    with tvm.transform.PassContext(opt_level=opt_level, required_pass=required_pass, disabled_pass=disabled_pass, config={"tir.add_lower_pass": [(3, save_original_pass)]}):
        lib = relay.build(model, target=target, params=params)

        if lib_params is not None:
            for lib_param in lib_params:
                lib.params[lib_param] = lib_params[lib_param]

        lib.export_library(script_dir + "/../../" + out_path_relative  + "/" + model_name + ".tar")
        print("Path: " + script_dir + "/../../" + out_path_relative  + "/" + model_name + ".tar")
        generate_model_end_code(lib, model, model_name, out_path_relative)


        output_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + ".json"
        
        with open(output_json_file, 'w') as outfile:
            print(lib.graph_json, file=outfile)
            outfile.close()

        print(lib.params.keys())
        output_params_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + "_lib_params.params"
        with open(output_params_json_file, 'wb') as outfile:
            outfile.write(relay.save_param_dict(lib.params))
            # print(relay.save_param_dict(params), file=outfile)
            outfile.close()

# Generate Relay IR Mutations.
def generate_relay_mutations(model, target, params, mutations, out_path_relative, opt_level=3, \
    required_pass=None, disabled_pass=None, quantize=False, opt_alias=""):

    model = relay.transform.InferType()(model)

    for mutation in mutations:

        mutation["metrics"] = {
            "call": 0,
            "total": 0
        }

        if(mutation["type"] == "change_layout"):

            # Load JSON config, regarding operation and layout transformations on data and kernel.
            desired_layouts = mutation['transformations']
            seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])
            model_mutation = seq(model)
            if ("title" in mutation):
                model_name = "model_" + mutation["title"]
            else:
                model_name = "change_layout_" + str(time.time())
    
        else:
            model_mutation = mutate_relay(model, mutation)
            if ("title" in mutation):
                model_name = "model_" + mutation["title"]
            else:
                model_name = "model_" + mutation["original_func_name"] + "_to_" + mutation["mutated_func_name"]

        model_name = model_name if not quantize else model_name + "_quant" 
        model_name = model_name + "_opt" + str(opt_level)
        model_name = model_name + (("_s" + str(mutation["start"])) if "start" in mutation else "")
        model_name = model_name + (("_e" + str(mutation["end"])) if "end" in mutation else "")
        model_name = model_name + opt_alias
        print("Generating model: " + model_name)

        os.makedirs(get_code_dir_path(model_name, out_path_relative), exist_ok=True)

        with tvm.transform.PassContext(opt_level=opt_level, required_pass=required_pass, disabled_pass=disabled_pass):
            lib = relay.build(model_mutation, target=target, params=params)
            
            if(mutation["metrics"]["call"] == 0):
                print ("Warning: Mutation had no effect in the graph. Skipping model generation...")
            else:
                lib.export_library(script_dir + "/../../" + out_path_relative  + "/" + model_name + ".tar")
                generate_model_end_code(lib, model_mutation, model_name, out_path_relative)

                output_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + ".json"
                with open(output_json_file, 'w') as outfile:
                    print(lib.graph_json, file=outfile)
                    outfile.close()

                print("Layer coverage (Relay):" + str(mutation["metrics"]["call"]) + "/" + str(mutation["metrics"]["total"]) + 
                "(" + str((mutation["metrics"]["call"]/mutation["metrics"]["total"])*100) + "%)")


# Generate TIR Mutations.
def generate_tir_mutations(model, target, params, mutations, out_path_relative, opt_level=3, required_pass=None, disabled_pass=None, quantize=False, opt_alias=""):
    
    # TODO: This is a hacky way of passing the params into the pass,
    # knowing that functions are first-class objects in python. Revise ASAP.

    for mutation in mutations:
        
        original_split = mutation["original"].rsplit('.', 1) if ("original") in mutation else "original"
        mutation_split = mutation["mutation"].rsplit('.', 1) if ("mutation") in mutation else "mutation"
        original_prettified =  original_split[1] if len(original_split) > 1 else mutation["original"]
        mutation_prettified = mutation_split[1] if len(mutation_split) > 1 else mutation["mutation"]
        model_name = "model_tir_" + original_prettified + "_to_" + mutation_prettified if ("title" not in mutation) else mutation["title"]
        model_name = model_name if not quantize else model_name + "_quant" 
        model_name = model_name + "_opt" + str(opt_level)
        model_name = model_name + (("_s" + str(mutation["start"])) if "start" in mutation else "")
        model_name = model_name + (("_e" + str(mutation["end"])) if "end" in mutation else "")
        model_name = model_name + opt_alias
        
        # Reset file.
        model_dir_path = get_code_dir_path(model_name, out_path_relative)
        os.makedirs(model_dir_path, exist_ok=True)
        output_tir_file = open(model_dir_path + "/output_tir.txt",'w').close()

        mutation["model_name"] = model_name
        mutation["out_path_relative"] = out_path_relative

        mutation["metrics"] = {
            "op": 0,
            "ops_affected": 0,
            "total": 0
        }
        mutate_tir.mutation_data = mutation
        
        print("Generating model: " + model_name)

        os.makedirs(get_code_dir_path(model_name, out_path_relative), exist_ok=True)

        with tvm.transform.PassContext(opt_level=opt_level, required_pass=required_pass, disabled_pass=disabled_pass, config={"tir.add_lower_pass": [(3, mutate_tir)]}):
            
            lib = relay.build(model, target=target, params=params)

            if(mutation["metrics"]["op"] == 0):
                print ("Warning: Mutation had no effect in the kernels. Skipping model generation...")

            else:
                # Export built model to .tar file.
                lib.export_library(script_dir + "/../../" + out_path_relative + "/" + model_name + ".tar")
                generate_model_end_code(lib, model, model_name, out_path_relative)

                output_json_file = script_dir + "/../../" + out_path_relative  + "/" + model_name + ".json"
                with open(output_json_file, 'w') as outfile:
                    print(lib.graph_json, file=outfile)
                    outfile.close()


                print("Op coverage (TIR):" + str(mutation["metrics"]["ops_affected"]) + "/" + str(mutation["metrics"]["total"]) + 
                "(" + str((mutation["metrics"]["ops_affected"]/mutation["metrics"]["total"])*100) + "%)")

def generate_model_end_code(lib, model, model_name, out_path_relative):

    output_host_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_host.txt",'w')
    output_device_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_device.txt",'w')
    output_relay_file = open(get_code_dir_path(model_name, out_path_relative) + "/output_relay.txt",'w')
    imported_modules = lib.get_lib().imported_modules

    if imported_modules is not None and len(imported_modules) > 0:
        dev_module = lib.get_lib().imported_modules[0]
        print(dev_module.get_source(), file=output_device_file)
        output_device_file.close()
    else:
        print("WARNING: Device code not generated for " + model_name + ".")
    print(lib.get_lib().get_source(), file=output_host_file)
    print(model, file=output_relay_file)
    output_host_file.close()
