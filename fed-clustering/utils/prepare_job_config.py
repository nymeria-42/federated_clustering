# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pathlib
import shutil

from nvflare.apis.fl_constant import JobConstants

from helpers import get_hash_trial

HASH_trial = get_hash_trial()

JOBS_ROOT = "jobs"


def job_config_args_parser():
    parser = argparse.ArgumentParser(
        description="generate train configs with data split"
    )
    parser.add_argument("--task_name", type=str, help="Task name for the config")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, help="Total number of sites")
    parser.add_argument(
        "--site_name_prefix", type=str, default="site-", help="Site name prefix"
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=0,
        help="Total data size, use if specified, in order to use partial data"
        "If not specified, use the full data size fetched from file.",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        help="Validation fraction of the total size, N = round(total_size* valid_frac), "
        "the first N to be treated as validation data. "
        "special case valid_frac = 1, where all data will be used"
        "in validation, e.g. for evaluating unsupervised clustering with known ground truth label.",
    )

    return parser

def _read_json(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} does not exist!")
    with open(filename, "r") as f:
        return json.load(f)


def _write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def _get_job_name(args) -> str:
    return args.task_name + "_" + str(args.site_num)


def _gen_deploy_map(num_sites: int, site_name_prefix: str) -> dict:
    deploy_map = {"app_server": ["server"]}
    for i in range(1, num_sites + 1):
        deploy_map[f"app_{site_name_prefix}{i}"] = [f"{site_name_prefix}{i}"]
    return deploy_map


def _update_meta(meta: dict, args):
    name = _get_job_name(args)
    meta["name"] = name
    meta["deploy_map"] = _gen_deploy_map(args.site_num, args.site_name_prefix)
    meta["min_clients"] = args.site_num


def _update_client_config(config: dict, args, site_name: str):
    # update client config
    # data path and training/validation row indices
    config["components"][0]["args"]["data_path"] = args.data_path
    config["components"][0]["args"]["valid_frac"] = args.valid_frac
    config["components"][0]["args"]["client_id"] = int(site_name.split("-")[-1])
    config["components"][0]["args"]["hash_trial"] = HASH_trial



def _update_server_config(config: dict, args):
    config["min_clients"] = args.site_num
    config["components"][3]["args"]["hash_trial"] = HASH_trial


def _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name):
    dst_path = dst_job_path / dst_app_name / "custom"
    os.makedirs(dst_path, exist_ok=True)
    src_path = src_job_path / src_app_name / "custom"
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def create_server_app(src_job_path, src_app_name, dst_job_path, site_name, args):
    dst_app_name = f"app_{site_name}"
    server_config = _read_json(
        src_job_path / src_app_name / "config" / JobConstants.SERVER_JOB_CONFIG
    )
    dst_config_path = dst_job_path / dst_app_name / "config"

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    _update_server_config(server_config, args)
    server_config_filename = dst_config_path / JobConstants.SERVER_JOB_CONFIG
    _write_json(server_config, server_config_filename)

    # copy custom file
    _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name)


def create_client_app(
    src_job_path, src_app_name, dst_job_path, site_name, args
):
    dst_app_name = f"app_{site_name}"
    client_config = _read_json(
        src_job_path / src_app_name / "config" / JobConstants.CLIENT_JOB_CONFIG
    )
    dst_config_path = dst_job_path / dst_app_name / "config"

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    # adjust file contents according to each job's specs
    _update_client_config(client_config, args, site_name)
    client_config_filename = dst_config_path / JobConstants.CLIENT_JOB_CONFIG
    _write_json(client_config, client_config_filename)

    # copy custom file
    _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name)


def main():
    parser = job_config_args_parser()
    args = parser.parse_args()
    job_name = _get_job_name(args)
    src_name = args.task_name + "_base"
    src_job_path = pathlib.Path(JOBS_ROOT) / src_name

    from dfa_lib_python.dataflow import Dataflow
    from dfa_lib_python.transformation import Transformation
    from dfa_lib_python.attribute import Attribute
    from dfa_lib_python.attribute_type import AttributeType
    from dfa_lib_python.set import Set
    from dfa_lib_python.set_type import SetType
    from dfa_lib_python.task import Task
    from dfa_lib_python.dependency import Dependency
    from dfa_lib_python.dataset import DataSet
    from dfa_lib_python.element import Element
    from dfa_lib_python.task_status import TaskStatus
    from dfa_lib_python.extractor_extension import ExtractorExtension
    from time import perf_counter

    dataflow_tag = "nvidiaflare-df"

    t2 = Task(2, dataflow_tag, "JobConfig")
    t2.begin()
    start = perf_counter()

    # create a new job
    dst_job_path = pathlib.Path(JOBS_ROOT) / job_name
    if not os.path.exists(dst_job_path):
        os.makedirs(dst_job_path)

    # update meta
    meta_config_dst = dst_job_path / JobConstants.META_FILE
    meta_config = _read_json(src_job_path / JobConstants.META_FILE)
    _update_meta(meta_config, args)
    _write_json(meta_config, meta_config_dst)

    # create server side app
    create_server_app(
        src_job_path=src_job_path,
        src_app_name="app",
        dst_job_path=dst_job_path,
        site_name="server",
        args=args,
    )

    # get num_rounds from the server config
    server_config = _read_json(
        dst_job_path / "app_server" / "config" / JobConstants.SERVER_JOB_CONFIG
    )
    args.num_rounds = server_config["num_rounds"]

    # create client side app
    for i in range(1, args.site_num + 1):
        create_client_app(
            src_job_path=src_job_path,
            src_app_name="app",
            dst_job_path=dst_job_path,
            site_name=f"{args.site_name_prefix}{i}",
            args=args,
        )

    to_dfanalyzer = [
        HASH_trial,
        args.task_name,
        args.data_path,
        args.site_num,
        args.site_name_prefix,
        args.data_size,
        args.valid_frac,
        args.num_rounds,
    ]
    t2_input = DataSet("iJobConfig", [Element(to_dfanalyzer)])
    t2.add_dataset(t2_input)
    t2_output = DataSet("oJobConfig", [Element([])])
    t2.add_dataset(t2_output)
    t2.end()


if __name__ == "__main__":
    main()
