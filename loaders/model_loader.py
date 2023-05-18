import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata

import tensorflow
import tensorflow.compat.v1 as tf

import onnx

class ModelLoader:

    def __init__(self):
        self.opset = 11

    def noop(self, data):
        return (None, None)

    # ------------ Loaders ------------
    def load_model(self, data, model_type=None):
        print ("Loading : " + data["name"] + " model.")

        if model_type == "remote":
            model_path = download_testdata(data["url"], data["alias"], module=data["type"])

        else:
            model_path = script_dir + "/" + data["url"] if data["url"].startswith("/") else data["url"]
        
        return self.load_onnx_model(model_path, data)

    def load_onnx_model(self, model_path, data, keep_dims=False, skip_inputs=False):
        print("Model Path: " + model_path)
        model = onnx.load(model_path)
        shape = data["input"]
        shape_dict = {}
        model = self._change_input_dim(model)
        print(data)
        if("library" in data and data["library"] is not None and "tflite" in data["library"] and keep_dims == False):
            di = shape
            shape = (di[0], di[3], di[2], di[1])
        # shape[0] = "unk__605"
        if(not skip_inputs):
            shape_dict[data["input_name"]] = shape

            return relay.frontend.from_onnx(model, shape_dict, dtype=data["dtype"], opset=self.opset, freeze_params=True)
        else:
            return relay.frontend.from_onnx(model, dtype=data["dtype"], opset=self.opset, freeze_params=True)


    # ------------ Helpers ------------

    # Function that replaces symbolic input dimension.

    def _change_input_dim(self, model):
        # Use some symbolic name not used for any other dimension.
        sym_batch_dim = "N"
        # or an actual value.
        actual_batch_dim = 1

        # The following code changes the first dimension of every input to be batch-dim
        # Modify as appropriate ... note that this requires all inputs to
        # have the same batch_dim.
        inputs = model.graph.input
        outputs = model.graph.output
        print("Inputs:")
        for input in inputs:
            print(input)
            
            dim = input.type.tensor_type.shape.dim

            i = 0
            while i < len(dim):
                dim1 = input.type.tensor_type.shape.dim[i]
                if("unk" in dim1.dim_param):
                    print ("Changed symbolic input dimension to actual for model.")
                    input.type.tensor_type.shape.dim[i].dim_value = 1
                    input.type.tensor_type.shape.dim[i].dim_param = "1"
                i += 1

        for output in outputs:
            
            dim = output.type.tensor_type.shape.dim

            i = 0
            while i < len(dim):
                dim1 = output.type.tensor_type.shape.dim[i]
                if("unk" in dim1.dim_param):
                    print ("Changed symbolic input dimension to actual for model.")
                    output.type.tensor_type.shape.dim[i].dim_value = 1
                    output.type.tensor_type.shape.dim[i].dim_param = "1"
                i += 1
 
        outputs = model.graph.output

        for output in outputs:

            # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
            # Add checks as needed.
            dim1 = output.type.tensor_type.shape.dim[0]
            if("unk" in dim1.dim_param or "dim" in dim1.dim_param):
                print (dim1)
                print ("Changed symbolic output dimension to actual for model.")
                output.type.tensor_type.shape.dim[0].dim_param = "1"

        return model
