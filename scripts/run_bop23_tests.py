import os
import os.path as osp
import argparse
import subprocess
import logging
from functools import partial
import multiprocessing
from tqdm import tqdm
import json

import numpy as np

from hydra import compose, initialize

from typing_extensions import Any, Dict, List, Optional, Tuple, Union, Sequence

# set level logging
logging.basicConfig(level=logging.INFO)


def run_inference_and_evaluation(dataset_name: str,
                                 name_prediction_file: str,
                                 run_config: List[str],
                                 dataset_dir: str,
                                 save_dir: str,
                                 with_evaluation: bool,
                                 evaluation_dir: str,
                                 overwrite: bool):

    result_file_path = osp.join(save_dir, "{}.json".format(name_prediction_file))
    if not osp.exists(result_file_path) or overwrite:
        # run inference
        inference_cmd = ["python", "-m", "scripts.run_inference"]
        config_cmd = ["dataset_name={}".format(dataset_name), "save_dir={}".format(save_dir),
                      "model.name_prediction_file={}".format(name_prediction_file)] + run_config
        inference_cmd.extend(config_cmd)

        logging.info("Run inference on the dataset <{}> with configuration: {}.".format(dataset_name, config_cmd))

        subprocess.run(inference_cmd, check=True)   # can take a while
        logging.info("Finished inference and saved the result in <{}>.".format(name_prediction_file))
    else:
        logging.info("Inference skipped because result file <{}> already exits.".format(result_file_path))

    if not with_evaluation:
        return None

    evaluation_file_path = osp.join(evaluation_dir, name_prediction_file, "scores_bop22_coco_segm.json")
    if not osp.exists(evaluation_file_path) or overwrite:
        # run evaluation
        evaluate_cmd = ["python", "bop_toolkit/scripts/eval_bop22_coco.py", "--results_path", save_dir,
                        "--result_filenames", "{}.json".format(name_prediction_file), "--eval_path", evaluation_dir,
                        "--datasets_path", dataset_dir, "--ann_type", "segm"]

        logging.info("Evaluate result of run: {}".format(name_prediction_file))
        subprocess.run(evaluate_cmd, check=True)
        logging.info("Evaluation finished.")
    else:
        logging.info("Evaluation skipped because result file <{}> already exits.".format(evaluation_file_path))

    # load and return evaluation data
    with open(evaluation_file_path, "r") as f:
        eval_data = json.load(f)

    return eval_data


def convert_dict_parameter_list(parameter_dict: Dict[str, str], seperator: str = "=") -> List[str]:
    """Convert '{name:value, ...}' to '[name=value, ...]'.

    :param parameter_dict:
    :param seperator:
    """
    parameter_list = [None] * len(parameter_dict)

    for (i, (param_name, param_value)) in enumerate(parameter_dict.items()):
        parameter_list[i] = "{}{}{}".format(param_name, seperator, param_value)

    return parameter_list


def convert_parameter_list_to_dict(parameter_list: Sequence[str], seperator: str = "=") -> Dict[str, str]:
    """Convert '[name=value, ...]' to '{name:value, ...}'.

    :param parameter_list:
    :param seperator:
    """
    parameter_dict = {}

    for param in parameter_list:
        param_name, param_value = param.split(seperator)[:2]
        parameter_dict[param_name] = param_value

    return parameter_dict


def convert_config_to_dict(configurations: Dict[str, Union[Dict[str, str], List[str]]]) -> Dict[str, str]:
    config_dict = {}

    for config_name, config_params in configurations.items():
        if not isinstance(config_params, dict):
            config_params = convert_parameter_list_to_dict(config_params)
        config_dict[config_name] = config_params

    return config_dict


def run_tests(dataset_names: Optional[List[str]],
              config_file_path: Optional[str],
              config_parameter_overrides: Optional[str],
              test_configuration_file: str,
              test_configuration_names: List[str],
              test_config_parameter_overrides: Optional[str],
              with_evaluation: bool,
              overwrite: bool) -> None:
    # load the hydra config file
    if config_file_path:
        config_path, config_name = osp.split(config_file_path)
    else:
        config_path = "../configs"
        config_name = "run_inference"

    with initialize(version_base=None, config_path=config_path, job_name="run_bop23_tests"):
        if config_parameter_overrides is not None:
            config_parameter_overrides = config_parameter_overrides.split(" ")
        cfg = compose(config_name=config_name, overrides=config_parameter_overrides)

    bop23_datasets = ["lmo",
                      "tless",
                      "tudl",
                      "icbin",
                      "itodd",
                      "hb",
                      "ycbv"]

    if dataset_names is None:
        datasets = bop23_datasets
    else:
        datasets = dataset_names

    # load test configurations
    with open(test_configuration_file, "r") as f:
        basic_run_configs = json.load(f)

    if len(test_configuration_names) > 0:
        basic_run_configs = {k: v for (k, v) in basic_run_configs.items() if k in test_configuration_names}

    # convert configuration to a consistent format
    basic_run_configs = convert_config_to_dict(basic_run_configs)

    # update basic configurations with parameter overrides
    if test_config_parameter_overrides:
        config_overrides = convert_parameter_list_to_dict(test_config_parameter_overrides)

        updated_run_configs = {}
        for (config_name, run_config) in basic_run_configs.items():
            updated_config = dict(run_config)
            updated_config.update(config_overrides)

            updated_run_configs[config_name] = updated_config
    else:
        updated_run_configs = basic_run_configs

    # convert parameter dict to list for execution
    updated_run_configs = {k: convert_dict_parameter_list(v) for (k, v) in updated_run_configs.items()}

    logging.info("Start inference and evaluation of the dataset: {}.".format(datasets))

    save_dir = cfg.save_dir
    evaluation_dir = osp.join(save_dir, "bop_eval")

    ds_names = []
    name_prediction_files = []
    run_configs = []
    evaluates = []
    for dataset_name in datasets:
        for (config_name, run_config) in updated_run_configs.items():
            ds_names.append(dataset_name)

            # structure 'method_datasetName-splitType'
            name_prediction_files.append(config_name + "_" + dataset_name + "-test")

            run_configs.append(run_config)

            # not every dataset has ground a truth available
            evaluates.append(with_evaluation and dataset_name in ["icbin", "lmo", "tless", "tudl", "ycbv"])

    # Note: partial doesn't work here because of the *args vs **kwargs mix
    def temp(ds_name: str, name_pred_file: str, config: List[str], evaluate: bool):
        nonlocal cfg, save_dir, evaluation_dir, overwrite
        return run_inference_and_evaluation(dataset_name=ds_name,
                                            name_prediction_file=name_pred_file,
                                            run_config=config,
                                            dataset_dir=cfg.data.root_dir,
                                            save_dir=save_dir,
                                            with_evaluation=evaluate,
                                            evaluation_dir=evaluation_dir,
                                            overwrite=overwrite)

    num_workers = 0#int(cfg.num_workers)
    if num_workers > 0:
        pool = multiprocessing.Pool(processes=num_workers)
        evaluation_results = list(tqdm(pool.imap_unordered(temp, ds_names, name_prediction_files, run_configs, evaluates),
                                  total=len(name_prediction_files)))
    else:
        evaluation_results = list(tqdm(map(temp, ds_names, name_prediction_files, run_configs, evaluates),
                                  total=len(name_prediction_files)))

    no_none_evaluation_results = [e for e in evaluation_results if e is not None]

    logging.info("Finished the inference of all datasets.")

    if with_evaluation and len(no_none_evaluation_results) > 0:
        # average results
        num_configs = len(basic_run_configs)

        keys = list(no_none_evaluation_results[0].keys())
        num_keys = len(keys)

        for (i, config_name) in enumerate(basic_run_configs.keys()):
            mean_result = np.zeros(num_keys, dtype=np.float32)

            non_none_counter = 0
            for j in range(i, len(evaluation_results), num_configs):
                dataset_evaluation = evaluation_results[j]

                if dataset_evaluation:
                    mean_result += np.array([dataset_evaluation[k] for k in keys])
                    non_none_counter += 1

            mean_result = mean_result / non_none_counter

            result_json = {keys[i]: float(v) for (i, v) in enumerate(mean_result)}

            result_file_path = osp.join(evaluation_dir, "{}_meta_result.json".format(config_name))
            with open(result_file_path, "w") as f:
                json.dump(result_json, f, indent=4)

            logging.info("Finished the evaluation of the <{}> configuration. The average result was saved as <{}> and is: {}".format(
                    config_name, result_file_path, result_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the inference and evaluation of the BOP23 datasets with multiple configurations.")

    parser.add_argument("-d", "--dataset_names", dest="dataset_names", type=str, nargs="+",
                        help="Perform inference and evaluation only on the given datasets.")

    parser.add_argument("-c", "--config_file_path", dest="config_file_path", type=str,
                        help="Path of the hydra configuration file.")
    parser.add_argument("-p", "--config_parameter_overrides", dest="config_parameter_overrides", type=str,
                        help="String of override parameters for hydra configuration file.")

    parser.add_argument("-r", "--test_configuration_file", dest="test_configuration_file", type=str, required=True,
                        help="Path of the '.json' file containing the all configuration for the tests.")
    parser.add_argument("-m", "--test_configuration_names", dest="test_configuration_names", type=str, required=True, nargs="+",
                        help="Name of the configurations to test for.")
    parser.add_argument("-n", "--test_config_parameter_overrides", dest="test_config_parameter_overrides", type=str,
                        help="String of override parameters for hydra configuration file used in all tests.")

    parser.add_argument("-e", "--no_evaluation", dest="no_evaluation", action="store_true", default=False,
                        help="If set no evaluation will be performed.")

    parser.add_argument("-o", "--no_overwrite", dest="no_overwrite", action="store_true", default=False,
                        help="If set, the old result and evaluation will be not overwritten.")

    args = parser.parse_args()

    run_tests(dataset_names=args.dataset_names,
              config_file_path=args.config_file_path,
              config_parameter_overrides=args.config_parameter_overrides,
              test_configuration_file=args.test_configuration_file,
              test_configuration_names=args.test_configuration_names,
              test_config_parameter_overrides=args.test_config_parameter_overrides,
              with_evaluation=not args.no_evaluation,
              overwrite=not args.no_overwrite)
