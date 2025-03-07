# MutateNN

This is the tool of the paper that was accepted and will appear in IEEE ICST Mutation 2025. You can access the paper preprint [here](https://arxiv.org/abs/2306.01697).

MutateNN is a comprehensive suite for mutation testing and analysis of DNNs utilized for image recognition tasks and deployed on a variety hardware acceleration devices. In total, the tool supports:
- Downloading and building Deep Neural Networks.
- Generating mutants from a variety of settings for testing purposes, given many parameterization capabilities.
- Generating device code for execution on different hardware acceleration devices.
- Executing inference on those devices for a dataset of inputs, following the necessary pre-processing, dependent on each DNN model.
- Performing analysis against all mutation configurations, for all devices, by supporting a variety of pairwise comparison operators, such as top-1, RBO and Kendall's Tau.
- Allowing debug execution and different optimizations applications on DNN models.


The suite utilizes [Apache TVM](https://tvm.apache.org/).

## Installation

The system needs TVM to be installed.
We also use `Python v3.x.x` (tested with `3.6.x`-`3.10.x`) and `Pip` as the package installer.

In addition, the system requires a number of pip packages. You can find them in the ./requirements.txt file

## Instructions

1. Install Python and Pip on your system.
- Python comes with linux distros usually, but this is not always the case for Pip. You can install it by running "sudo apt install python3-pip"
2. Download and install TVM:
For instructions of how to install TVM, please refer to the [TVM Installation Guide](https://tvm.apache.org/docs/install/index.html), or check [TVM related guide for developers](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github).
We built the system using the TVM, v0.13.0. Newer versions are untested at the moment.

Follow the installation from source instructions, and based on the experiments you want to run, enable the respective flags in the `<tvm_folder>/build/config.cmake`. For our experiments, we followed different settings per-device, but consider enabling the `USE_LLVM` and `USE_OPENCL` or `USE_CUDA` flags, depending on your configuration.

3. Install necessary packages by executing the command:
`pip install -r requirements.txt` 

4. Download necessary models, if you wish to run them locally. Alternative, you can instruct the MutateNN to download them for you.
Although system utilizes already provided models for Keras and PyTorch, we utilized some TF/TFlite models from the GitHub repo of Tensorflow for slim Models, such as:
- `MobileNetV2`
- `ResNet152`
- `InceptionV2`

You can download them manually and place them in the models folder of each model from [the official TensorFlow repo](https://github.com/tensorflow/models/tree/master/research/slim).


## Configuration
The configuration of the system is included into the `config.json` file.
Each section is self-explanatory and defines which part it concerns.
Important notes:
- You can utilize the TVM debugger, by setting `debug_enabled: true`.
- You can set different TVM optimization settings, by modifying the `opt_level` variable from 0 to 4, with optimization level increasing.
- You can set the mutations you want to generate, by modifying the `mutations` entry of the object.
- You can specify the occurence numbers that you want your mutation to be applied, by modifying `positions` parameter in mutations. You can see examples of it on the configuration file provided.
- Device settings have been cleared out to preserve anonymity. If you wish, you can set up your own TVM RPC server on your own device and run everything following the instructions [here](
https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html).

## Example
In order to verify your installation and be able to run the framework with your own configuration, we have setup the configuration to build the system utilizing 7 models under test: `MobileNetV2, DenseNet121, ResNet152, AlexNet, ShuffleNet, EfficientNetLite and InceptionV2`.
You can download, run and evaluate the models accordingly. All models are obtained from the slim official repository, are pre-trained against `ImageNet` and perform classification tasks, for 1000 [labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

We also provide a small test dataset, consisting of 5 public domain images (Sample Dataset Images obtained from [unsplash](https://unsplash.com/images/stock/public-domain). No copyright infingement intended.) To demonstrate device comparison, we have generated 3 simulations on different devices for `MobileNetV2`, which can be found on `/generated/MobileNetV2/simple_run`.
 You can instruct `MutateNN` to build, run and evaluate the existing dataset against these device outputs, by setting `build, execute` and `evaluate` to `true` in `MobileNetV2` entry on `config.json`.

 Each model configuration entry, contains a number of necessary parameters, such as the input layer name and size, the output layer, etc.


Once you set up the framework, you can execute it by doing:
`python3 main.py`

#### Model Build/Mutants Generation:

Inside `config.json`, you can set the mutations you want to generate, by modifying the `mutations` entry of the object. You can instruct `MutateNN` to generate mutants on the model's graph, by using `Relay IR`, or in the Tensor level IR, using `TIR`. A number of supported mutations are already provided, but they can be modified and parameterized, based on the user needs.


The system will generate the models in the folder defined in config.json, along with their generated Host/Kernel code, but also their TVM Relay IR code:
`<script_folder>/<models_out_relative>//<model_variant>_<opt_setting>_code`

In total, the framework will generate the models compiled on TVM, utilizing the `opt=2` optimization setting by default.

You can also try your own model, given you provide the right files and settings. Configuration provides exactly the details requested for a model to be loaded from a backend, compiled using a specific optimization and GPU backend and be run for inference, respectfully.


#### Execution/Inference
Your system will then execute, generating a folder with experiments. The structure followed is:
Build: `<script_folder>/<models_out_relative>//<model_variant>_<opt_setting>.tar`
Execution: `<script_folder>/<exec_out_relative>/mutations/ts_<epoch_time_of_run>/<predictions>.txt`
Evaluation: `<script_folder>/<evaluation_out_relative>/`

Based on existing configuration, inference generates the top-5 predictions, along with the execution time per-prediction at the bottom.
In addition, you will find an execution_log.txt file in the aforementioned folder, containing info about the run.

Console will indicate the status of the running model and update accordingly.

#### Analysis
Once execution is complete, analysis will be executed. This will be done in 3 ways:
- Comparing results per-device (if provided), in JSON files.
- Separate csv and json files, containing metadata related to the execution.
- Comparing results per-multiple executions (if provided)

The system will then generate the following files:
`<evaluation_folder>/device_evaluation.json` containing results per-device comparison in a pairwise manner.
`<evaluation_folder>/device_discrepancies.json` containing only the cases where dissimilarities are observed.

The system performs analysis in an intelligent manner - it performs folder exploration to determine discrepancies and if it identifies a crashed execution bundle, or a case where the network has performed with heavy degradation (e.g., producing many similar results for different inputs), then it terminates.


#### Errors
In case of an error, the suite will generate a `<script_folder>/error_log/<model>/ts_<epoch_time_of_problematic_run>/error_log.txt` file containing all related info.

<!--## Demonstration
You can watch a tool demonstration on [YouTube](https://www.youtube.com/watch?v=j7Ffd5y_i3g).-->
