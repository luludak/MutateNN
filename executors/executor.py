import tvm
import tvm.runtime as runtime
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.contrib import utils

import time
import math
import os
import json
import traceback

import numpy as np
from scipy.special import softmax

from os import listdir
from os.path import isfile, join

from PIL import Image

from processors.model_preprocessor import ModelPreprocessor
from tvm.contrib.debugger.debug_executor import GraphModuleDebug

from helpers import timer

class Executor:
    def __init__(self, models_data, images_data, connection_data, extra_folder, image_count_threshold = 0):
        self.name = models_data["name"]
        self.model_name = models_data["model_name"]
        self.converted_lib_model_name = models_data["converted_lib_model_name"] if "converted_lib_model_name" in models_data else None
        self.raw_model_name = models_data["raw_model_name"]
        self.input_model_folder = models_data["input_model_folder"]
        self.output_model_folder = models_data["output_model_folder"]
        self.mutations_names = models_data["mutations_names"]
        self.image_dimension = models_data["image_dimension"]
        self.input = models_data["input"]
        self.input_name = models_data["input_name"]
        self.output = models_data["output"]
        self.output_name = models_data["output_name"]
        self.library = models_data["library"] if "library" in models_data else None
        self.input_images_folders = images_data["input_images_folders"]
        self.output_images_base_folder = images_data["output_images_base_folder"]
        self.extra_folder = extra_folder
        self.connection_id = "local"
        self.connection_data = connection_data
        self.error_base_folder = connection_data["error_base_folder"]
        self.preprocessing_enabled = models_data["preprocessing_enabled"]
        self.debug_enabled = models_data["debug_enabled"]
        self.op_type =  models_data["op_type"] if "op_type" in models_data else None
        self.params = models_data["params"] if "params" in models_data else None
        self.image_count_threshold = image_count_threshold

        self.module = None
        self.device = None
        self.loaded_module = None
        self.json_str = None

        self.timer = timer.Timer()

    def prepare(self, variant, remote = None):

        device_id = self.connection_data["id"] or 0
        host_type = self.connection_data["host_type"]
        if(host_type == "local_no_rpc" or host_type == "local" or remote == None):
            self.loaded_module = runtime.module.load_module(self.input_model_folder + variant)
            target_framework = self.connection_data["target_framework"]
            self.device = tvm.device(str(target_framework), device_id)
        else:

            model_path = join(self.input_model_folder, variant)
            remote.upload(model_path)
            target_framework = self.connection_data["target_framework"]
            self.loaded_module = remote.load_module(variant)
            self.device = remote.device(target_framework, device_id)

        
        if (self.module is None):
            if (not self.debug_enabled):
                self.module = graph_executor.GraphModule(self.loaded_module["default"](self.device))
            else:
                # Prepare graph.
                with open(self.input_model_folder + variant.replace(".tar", ".json")) as f:
                    self.json_str = f.read()
                    f.close()

                    self.module = GraphModuleDebug(
                        self.loaded_module["debug_create"]("default", self.device),
                        [self.device],
                        self.json_str,
                        dump_root=self.output_images_base_folder + "/" + (self.library if self.library else "") + "/debug/")
            
        return self

    def execute(self):
        if(self.module == None or self.device == None):
            raise Exception("Error: Device not initialized.")
        return "ts_0"

    def extract_image_names(self, folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    def process_images(self):
        self.process_images_with_output(module, self.input_images_folder, self.output_images_folder)
        return self

    def reset_module(self, variant):
            
        self.module = GraphModuleDebug(
            self.loaded_module["debug_create"]("default", self.device),
            [self.device],
            self.json_str,
            dump_root=self.output_images_base_folder + "/" + (self.library if self.library else "") + "/debug/")
        print("----- Module Reset. -----")

    def process_images_with_io(self, input_folder, output_folder, model_name=None, variant=None):    

        if (model_name == None):
            model_name = self.model_name

        os.makedirs(output_folder, exist_ok=True)

        print("Processing Dataset images in folder: " + input_folder)

        image_names = self.extract_image_names(input_folder)

        image_names_length = len(image_names)
        image_length = (image_names_length - 1) if (image_names_length > 1) else image_names_length
        step = (image_length // 4) if image_length > 4 else image_length
        
        count = 0
        same_ranks = 0
        prev_ranks = None

        input_shape_inferred = self.module.get_input(0).shape

        data = {
            "input": input_shape_inferred,
            "image_dimension": self.image_dimension,
            "library": self.library
        }
        model_preprocessor = ModelPreprocessor(data)
        error_occured = False
        errors_no = 0
        for image_name in image_names:

            if(count % step == 0):
                print("Complete: " + str((count // step) * 25) + "%")

            count += 1
            if (image_name.startswith(".")):
                print("Skipping " + image_name)
                continue

            img_path = input_folder + "/" + image_name
           
            img = Image.open(img_path)

            # Convert monochrome to RGB
            if(img.mode == "L"):
                img = img.convert("RGB")

            img = img.resize(self.image_dimension)
            name_to_use = self.converted_lib_model_name if self.converted_lib_model_name is not None else self.model_name

            
            img = model_preprocessor.preprocess(name_to_use, img, self.preprocessing_enabled)

            image_name_extracted = image_name.rsplit('.', 1)[0]
        
            output_file_path = output_folder + "/" + image_name_extracted.replace(".", "_") + ".txt"
            
            try:
                start_timestamp = self.timer.get_epoch_timestamp(False)
                self.module.set_input(0, img)
                if (self.params is not None):
                    self.module.load_params(relay.save_param_dict(self.params))
                self.module.run()

                out = tvm.nd.empty(self.output, device=self.device)
                tvm_output = self.module.get_output(0, out).numpy()

                end_timestamp = self.timer.get_epoch_timestamp(False)
                

                scores = softmax(tvm_output)
                if(len(scores) > 2):
                    squeeze(scores, [0, 2])

                scores = np.squeeze(scores)
                ranks = np.argsort(scores)[::-1]
                extracted_ranks = ranks[0:5]

                # Note: this is not the most optimal way as it misses if there are previous occurences
                # before the N-1 occurence. However, in practice, it is proven sufficient to detect
                # problematic executions.
                if(prev_ranks is not None and np.array_equal(prev_ranks, extracted_ranks)):
                    same_ranks += 1
                else:
                    prev_ranks = extracted_ranks
                    same_ranks = 0

                if(self.image_count_threshold != 0 and same_ranks == self.image_count_threshold):
                    print ("Warning: Execution produced same ranks for 100 consecutive inputs. Stopped.")
                    return self

                with open(output_file_path, 'w') as output_file:
                    # TOP-K of ranks, with K=5.
                    for rank in extracted_ranks:
                        print("%s, %f" % (rank, scores[rank]), file = output_file)
                    
                    print("\n%f" % (end_timestamp - start_timestamp), file = output_file)
                    output_file.close()

                if(self.debug_enabled):
                    debug_base = self.output_images_base_folder + "/" + self.library + "/debug/_tvmdbg_device_OPENCL_0/"
                    os.rename(debug_base + "/_tvmdbg_execution_trace.json", debug_base + "/_tvmdbg_execution_trace_" + image_name.split(".")[0] + ".json")
                    os.rename(debug_base + "/output_tensors.params", debug_base + "/output_tensors_" + image_name.split(".")[0] + ".params")
                    self.reset_module(variant)

            except Exception as e:
                ts_floor = str(math.floor(self.timer.get_epoch_timestamp(False)))
                errors_no += 1
                if(not error_occured):
                        error_occured = True
                        print("One or more image errors have occured. See execution log of the model for details (ts: " + ts_floor + ").")
                
                if(errors_no == 100):
                    print ("100 Errors have occured. Stopping execution.")
                    return

                folder_to_write = join(self.error_base_folder, ts_floor)
                os.makedirs(folder_to_write, exist_ok=True)
                f = open(join(folder_to_write, 'error_log.txt'), 'a+')
                f.write("Model: " + model_name + "\nOutput folder:" + output_folder + "\nImage: "+ img_path + "\n")
                traceback.print_exc(file=f)
                f.close()

        return self