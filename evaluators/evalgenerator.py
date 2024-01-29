
import os
from os import path
from os import listdir
from os.path import isfile, isdir, join, exists

import fnmatch

import statistics
from scipy import stats
import math
import numpy as np

import json
import csv

from evaluators.classification import Evaluator

class EvaluationGenerator:

    # TODO: Refactor Args.
    def __init__(self, out_path=None, original_model_name="model_original", 
    build_ts = "ts_1", exec_log_prefix = "execution_log", evaluate_log_prefix="evaluate", image_count_threshold=0):
        self.evaluator = Evaluator()
        self.exec_log_prefix = exec_log_prefix
        self.evaluate_log_prefix = evaluate_log_prefix
        self.difference_tolerance = 5
        self.comparison_heuristic = 0.7
        self.image_count_threshold = image_count_threshold
        
        
    def get_same_folder_comparison(self, base_folder, base_case=None):

        evaluation_folders  = [d for d in listdir(base_folder) if isdir(join(base_folder, d))]
        # print("Evaluating folder " + base_folder + ".")
        if evaluation_folders == None or len(evaluation_folders) == 0:
            return

        evaluation_folders = sorted(evaluation_folders)
        
        # evaluation_folders = evaluation_folders_sorted[-10:] if (len(evaluation_folders_sorted) >= 10) else evaluation_folders_sorted
        comparisons = []
        times = {}
        total_images_dissimilar = {}
        total_avg_exec_time = 0

        base_folders = evaluation_folders if base_case is None else [base_case]

        for base in base_folders:

            for evaluated in evaluation_folders:

                if (base == evaluated):
                    continue
                
                comparison_stats = self.get_basic_evaluation(join(base_folder, base, "mutations"), join(base_folder, evaluated, "mutations"), base_folder, False)
                # print(evaluated)
                # print(comparison_stats)
                if(comparison_stats is None):
                    continue

                evaluated_avg_time = comparison_stats["comparison"]["average_exec_time"]
                evaluated_total_time = comparison_stats["comparison"]["total_exec_time"]
                images_dissimilar = comparison_stats["comparison"]["images_dissimilar"]
                del comparison_stats["comparison"]["average_exec_time"]
                del comparison_stats["comparison"]["total_exec_time"]
                if(evaluated not in times):
                    times[evaluated] = {
                        "average_exec_time": evaluated_avg_time,
                        "total_exec_time": evaluated_total_time
                    }
                    total_avg_exec_time += evaluated_avg_time

                comparisons.append({
                    "base": base,
                    "evaluated": evaluated,
                    "comparison": comparison_stats["comparison"]
                })

                for image_dissimilar in images_dissimilar:
                    if image_dissimilar not in total_images_dissimilar:
                        total_images_dissimilar[image_dissimilar] = 0
                    else:
                        total_images_dissimilar[image_dissimilar] += 1 

        evaluation_folders_len = len(evaluation_folders) if len(evaluation_folders) > 0 else 1
        full_object = {
            "times": times,
            "total_avg_exec_time": total_avg_exec_time / evaluation_folders_len,
            "comparisons": comparisons,
            "total_images_dissimilar": total_images_dissimilar
        }
        
        with open(join(base_folder, "same_folder.json"), 'w') as outfile:
            print(json.dumps(full_object, indent=2, sort_keys=True), file=outfile)
            outfile.close()

        return full_object

    def distinguish_device_discrepancies(self, base_folder):

        new_obj = {}
        with open(join(base_folder, "device_evaluation.json")) as eval_file:

            eval_dict = json.load(eval_file)

            for mutation in eval_dict.keys():

                mutation_data = eval_dict[mutation]
                new_data = []

                if (mutation_data == []):
                    new_data.append({
                        "type": "full_crash"
                    })
                    continue

                for comparison in mutation_data:
                    
                    comparison_data = comparison["comparison"] if "comparison" in comparison else {}
                    if (comparison_data == {}):
                        continue

                    if(int(comparison_data["total_no_of_images"]) > self.image_count_threshold and float(comparison_data["percentage_dissimilar"]) > 0):
                        new_data.append({
                            "base": comparison["base"],
                            "evaluated": comparison["evaluated"],
                            "percentage_dissimilar": comparison_data["percentage_dissimilar"],
                            "type": "discrepancy"
                        })
                    
                    elif (int(comparison_data["total_no_of_images"] <= self.image_count_threshold)):
                        new_data.append({
                            "base": comparison["base"],
                            "evaluated": comparison["evaluated"],
                            "no_of_images": comparison_data["total_no_of_images"],
                            "type": "crash"
                        })


                new_obj[mutation] = new_data

        with open(join(base_folder, "device_discrepancies.json"), 'w') as outfile:
            print(json.dumps(new_obj, indent=2, sort_keys=True), file=outfile)
            outfile.close()



    def generate_devices_comparison(self, base_folder, replace_evaluated_suffix=False):

        devices = [d for d in listdir(base_folder) if isdir(join(base_folder, d))]

        comparisons = {}
        times = {}
        total_images_dissimilar_across_devices = {}
        total_csv_obj = []

        print("Generating Comparison for " + base_folder)

        for base in devices:

            libraries = [l for l in listdir(join(base_folder, base)) if isdir(join(base_folder, base, l))]

            for library in libraries:

                if(not library in comparisons):
                    comparisons[library] = []

                base_lib_path = join(base_folder, base, library)
                if not base_lib_path.endswith("mutations"):
                    base_lib_path = join(base_lib_path, "mutations")

                if (not exists(base_lib_path)):
                    print("Base Path does not exist. Skipping.....")
                    continue

                for evaluated in devices:

                    evaluated_lib_path = join(base_folder, evaluated, library)

                    if not evaluated_lib_path.endswith("mutations"):
                        evaluated_lib_path = join(evaluated_lib_path, "mutations")

                    if (base == evaluated):
                        continue

                    elif (not isdir(evaluated_lib_path)):
                        print("Evaluated mutations folder does not exist. Skipping.....")
                        continue

                    comparison_stats = self.get_basic_evaluation(base_lib_path, evaluated_lib_path, base_folder, False)
                    
                    if(comparison_stats is None):
                        continue

                    images_dissimilar = comparison_stats["comparison"]["images_dissimilar"]
                    for image_dissimilar in images_dissimilar:
                        if image_dissimilar not in total_images_dissimilar_across_devices:
                            total_images_dissimilar_across_devices[image_dissimilar] = 0
                        else:
                            total_images_dissimilar_across_devices[image_dissimilar] += 1 

                    del comparison_stats["comparison"]["exec_time_percentages"]

                    comparisons[library].append({
                        "base": base,
                        "evaluated": evaluated,
                        "comparison": comparison_stats["comparison"]
                    })
                    
                    stats = comparison_stats["comparison"]["oneway_exec_time"]
                    if stats["p-value"] is not None and float(stats["p-value"]) > 0.05:
                        
                        total_csv_obj.append({
                            "lib": library,
                            "device1" : base,
                            "device2": evaluated,
                            "p-value" : stats["p-value"],
                            "statistic" : stats["statistic"]
                        })
        
        comparisons["total_images_dissimilar"] = total_images_dissimilar_across_devices

        output_path_file = base_folder + "/device_evaluation.json"
        
        type_under_test = base_folder.split("/")[-1] if not base_folder.endswith("/") else base_folder.split("/")[-2]    

        output_stats_total_csv = base_folder + "/" + type_under_test + "_" + "output_stats_total_1.csv"

        with open(output_path_file, 'w') as outfile:
            print(json.dumps(comparisons, indent=2, sort_keys=True), file=outfile)
            outfile.close()

        csv_fields = ['lib', 'device1', 'device2', 'p-value', 'statistic']

        with open(output_stats_total_csv, 'a', newline='') as file: 
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writerows(total_csv_obj)


    def get_basic_evaluation(self, folder1_path, folder2_path, output_path_file, include_individual_analysis=True, write_to_file=False, dissimilar_images_max_no=10, max_no_of_diff_labels=0, verbose_time_data=False):

        if(not os.path.exists(folder1_path) or not os.path.exists(folder2_path)):
            print ("Original path does not exist. Skipping.")
            return
        folder1_ts_folder_name = [fo for fo in listdir(folder1_path) if not isfile(join(folder1_path, fo)) and "ts_" in fo]
        if (len(folder1_ts_folder_name) == 0):
            print ("Source TS path invalid. Skipping...")
            return        
        folder1_ts_folder_name = folder1_ts_folder_name[0]

        folder2_ts_folder_name = [fo for fo in listdir(folder2_path) if not isfile(join(folder2_path, fo)) and "ts_" in fo]
        if (len(folder2_ts_folder_name) == 0):
            print ("Target TS path invalid. Skipping...")
            return
        
        folder2_ts_folder_name = folder2_ts_folder_name[0]

        original_model_path = join(folder1_path, folder1_ts_folder_name)
        mutation_model_path = join(folder2_path, folder2_ts_folder_name)

        # Size matching checks to avoid useless computations
        # on crashed models.
        if(not os.path.exists(original_model_path)):
            print("Warning: original path " + original_model_path + " does not exist. Skipping evaluation.....")
            return
        elif(not os.path.exists(mutation_model_path)):
            print("Warning: mutation path " + mutation_model_path + " does not exist. Skipping evaluation.....")
            return

        mut_no = len(fnmatch.filter(os.listdir(mutation_model_path), '*.txt')) 
        orig_no = len(fnmatch.filter(os.listdir(original_model_path), '*.txt'))

        if(abs(mut_no - orig_no) >= self.difference_tolerance):
            print("Warning: " + mutation_model_path + " contains different number of files than base folder. Skipping evaluation.....")
            return
            

        evaluation_data_obj = {}

        image_txt_names  = [f for f in listdir(original_model_path)
                            if isfile(join(original_model_path, f)) and f.endswith(".txt") and
                            f not in output_path_file and self.evaluate_log_prefix not in f and self.exec_log_prefix not in f and "Error" not in f]
        
        total_images_no = len(image_txt_names)
        images_similar_no = 0
        images_dissimilar = 0
        images_dissimilar_no = 0


        paths_to_check = [join(mutation_model_path, f) for f in image_txt_names]
        files_no_diff = abs(len(image_txt_names) - len(paths_to_check))

        # Check that NN is not "stuck" - producing the same results in every input.
        # We do this check by applying comparisons of the first 100 sample results.
        problematic_occurences = self._count_problematic_occurences(paths_to_check[0], paths_to_check[1:101])\
            if len(paths_to_check) > 100 else len(paths_to_check)
            
        if(len(paths_to_check) > 100 and problematic_occurences > 90):
            output_error_file = join(path.dirname(output_path_file), "Error.txt")
            error_text = mutation_model_path + " was found " + str(problematic_occurences) + "% problematic. Skipped generation of evaluation file.\n"
            error_text += mutation_model_path + " was found to have a different number of files by " + str(files_no_diff) + "."
            print(error_text)
            with open(output_error_file, 'w') as outfile:
                print (error_text, file=outfile)

                outfile.close()
            return
        total_exec_time = 0.0
        orig_total_exec_time = 0.0
        images_dissimilar = []
        exec_time_percentages = []


        total_count_of_labels = {}
        diff_labels = {}


        orig_exec_times = []
        mut_exec_times = []
        
        
        for image_txt in image_txt_names:

            original_img_file_path = join(original_model_path, image_txt)
            mutation_img_file_path = join(mutation_model_path, image_txt)
            
            if not path.isfile(mutation_img_file_path):
                print("- Warning: File " + mutation_img_file_path + " does not exist in model folder. Skipping...")
                continue

            image_name_extracted = image_txt.split('.')[0]
            evaluated = self.evaluator.evaluate(original_img_file_path, mutation_img_file_path)
            
            orig_exec_time = evaluated["base_exec_time"]
            orig_exec_times.append(orig_exec_time)
            mut_exec_time = evaluated["exec_time"]
            mut_exec_times.append(mut_exec_time)
            
            total_exec_time += mut_exec_time
            orig_total_exec_time += orig_exec_time
            comparison = float(evaluated["comparisons"]["first_only"])
            exec_percentage = ((mut_exec_time - orig_exec_time) / (orig_exec_time)) * 100
            exec_time_percentages.append(round(exec_percentage, 2))

            # TODO: Change threshold
            if evaluated["base_label1"] not in total_count_of_labels:
                total_count_of_labels[evaluated["base_label1"]] = 1
            else:
                total_count_of_labels[evaluated["base_label1"]] += 1

            # For comparisons other than first_only: we use a heuristic threshold
            # to consider that two images are similar.
            if (comparison >= self.comparison_heuristic):
                images_similar_no += 1
            else:
                images_dissimilar_no += 1
                if (evaluated["base_label1"] not in diff_labels):
                    diff_labels[evaluated["base_label1"]] = 1
                else:
                    diff_labels[evaluated["base_label1"]] += 1

                # Set flag to -1 in order to include all dissimilar images to analysis.
                if(dissimilar_images_max_no == -1 or images_dissimilar_no < dissimilar_images_max_no):
                    images_dissimilar.append(image_txt)

            if(include_individual_analysis):
                evaluation_data_obj[image_name_extracted] = evaluated

        div_total_images_no = total_images_no if total_images_no > 0 else 1

        diff_labels = dict(sorted(diff_labels.items(), key=lambda item: item[1], reverse=True)[0:max_no_of_diff_labels])
 
        total_diff_label_info = {}
        for key in diff_labels:
            different = diff_labels[key]
            total_diff_label_info[key] = {
                "different": different,
                "total": total_count_of_labels[key],
                "percentage": (different/total_count_of_labels[key])*100
            }

        # Calculate One-way ANOVA for execution times - to verify statistical significance.
        t_test_result = stats.f_oneway(orig_exec_times, mut_exec_times)        

        evaluation_data_obj["comparison"] = {
            "total_no_of_images": total_images_no,
            "no_of_images_similar": str(images_similar_no),
            "no_of_images_dissimilar": str((images_dissimilar_no)),
            "percentage_similar": str((images_similar_no / (div_total_images_no)) * 100),
            "percentage_dissimilar": str((images_dissimilar_no / div_total_images_no) * 100),
            "total_exec_time": total_exec_time,
            "orig_total_exec_time": orig_total_exec_time,
            "average_exec_time": total_exec_time / div_total_images_no,
            "average_orig_exec_time": orig_total_exec_time / div_total_images_no,
            "images_dissimilar": images_dissimilar,
            "exec_time_percentages": exec_time_percentages,
            "diff_labels": total_diff_label_info,
            "oneway_exec_time": {
                "statistic": t_test_result[0] if not math.isnan(t_test_result[0]) else "NaN",
                "p-value": t_test_result[1] if not math.isnan(t_test_result[1]) else "NaN"
            }
        }

        if (verbose_time_data):
            evaluation_data_obj["comparison"]["times"] = {
                "orig_exec_times" : orig_exec_times,
                "mut_exec_times" : mut_exec_times
            }

        if(write_to_file):
            with open(output_path_file, 'w') as outfile:
                print(json.dumps(evaluation_data_obj, indent=2, sort_keys=True), file=outfile)
                outfile.close()

        return evaluation_data_obj


    def _count_problematic_occurences(self, original_path, samples_paths):

        problematic_count = 0
        for sample_path in samples_paths:
            if(self.evaluator.compare_to_original(original_path, sample_path)):
               problematic_count += 1 
        
        # If problematic occurences over threshold, then mark as problematic.
        return problematic_count
