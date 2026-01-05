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

from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range, load_data
from nvflare.app_common.app_constant import AppConstants

from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.dependency import Dependency

from time import perf_counter
import datetime
from pathlib import Path

dataflow_tag = "nvidiaflare-df"


def ensure_serializable(obj):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class DBSCANLearner(Learner):
    def __init__(
        self,
        data_path: str,
        client_id: int = 0,
        hash_trial: str = "unknown_trial",
        valid_frac: float = 0.2,
        eps: float = 0.5,  # Maximum distance between samples to be considered neighbors
        min_samples: int = 5,  # Minimum samples in neighborhood for core point
        random_state: int = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.valid_frac = valid_frac
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.client_id = int(client_id)
        self.train_data = None
        self.valid_data = None
        # number of training samples (used by NVFlare SKLearnExecutor)
        self.n_samples = None
        self.hash_trial = hash_trial

    def load_data(self) -> dict:
        t3 = Task(3, dataflow_tag, "LoadData")
        t3.begin()
        start = perf_counter()

        train_data = load_data(self.data_path, require_header=True)
        data_size = train_data[-1]
        valid_size = int(round(data_size * self.valid_frac))

        indices = {
            "valid": {"start": 0, "end": valid_size},
        }
        
        train_data = load_data(self.data_path)
        valid_data = load_data_for_range(
            self.data_path,  
            indices["valid"]["start"],
            indices["valid"]["end"],
        )   

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t3_input = DataSet("iLoadData", [Element(to_dfanalyzer)])
        t3.add_dataset(t3_input)
        t3_output = DataSet("oLoadData", [Element([])])
        t3.add_dataset(t3_output)
        t3.end()

        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]

        t4 = Task(4, dataflow_tag, "InitializeClient",
                  dependency=Task(3, dataflow_tag, "LoadData"))
        t4.begin()
        start = perf_counter()
        duration = perf_counter() - start

        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t4_input = DataSet("iInitializeClient", [Element(to_dfanalyzer)])
        t4.add_dataset(t4_input)
        t4_output = DataSet("oInitializeClient", [Element([])])
        t4.add_dataset(t4_output)
        t4.end()

        # set number of samples for compatibility with SKLearnExecutor
        # load_data returns a tuple like (x, y, n_samples)
        try:
            self.n_samples = data["train"][-1]
        except Exception:
            # fallback: compute from x_train if available
            try:
                x_train = self.train_data[0]
                self.n_samples = len(x_train)
            except Exception:
                self.n_samples = 0

    def train(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        t5 = Task(5 + 4 * (curr_round), dataflow_tag, "ClientTraining")
        if curr_round == 0:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", "0"],
                )
            )
        else:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", str(8 + 4 * (curr_round - 2))],
                )
            )

        t5.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        
        # Update hyperparameters from global if provided
        if global_param:
            self.eps = global_param.get("eps", self.eps)
            self.min_samples = global_param.get("min_samples", self.min_samples)

        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            self.eps,
            self.min_samples,
            timestamp
        ]

        t5_input = DataSet("iClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_input)

        # Get training data and perform local DBSCAN
        (x_train, y_train, train_size) = self.train_data
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        dbscan.fit(x_train)

        # Extract core points and their labels
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        
        core_points = x_train[core_mask]
        core_labels = dbscan.labels_[core_mask]

        saved_path = None
        try:
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            safe_hash = str(self.hash_trial).replace("/", "_")[:32]
            filename = f"dbscan_client{self.client_id}_{safe_hash}.npz"
            file_path = artifacts_dir / filename

            # Ensure numpy arrays for saving
            cp_arr = np.array(core_points)
            cl_arr = np.array(core_labels)
            np.savez_compressed(str(file_path), core_points=cp_arr, core_labels=cl_arr)
            saved_path = str(file_path)
        except Exception as e:
            try:
                self.log_error(fl_ctx, f"dbscan learner: failed to save artifact: {e}")
            except Exception:
                pass
        # Prepare return parameters - only core points and their labels
        # Ensure all numpy types are converted to Python native types for serialization
        params = {
            "core_points": core_points.tolist() if isinstance(core_points, np.ndarray) else list(core_points),
            "core_labels": [int(x) for x in core_labels.tolist()] if isinstance(core_labels, np.ndarray) else [int(x) for x in core_labels],
            "eps": float(self.eps),
            "min_samples": int(self.min_samples),
            "n_clusters": int(len(set(core_labels)) - (1 if -1 in core_labels else 0))
        }

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            saved_path,
            params["n_clusters"],
            len(core_points),
            duration,
            timestamp
        ]

        t5_output = DataSet("oClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_output)
        t5.end()

        # Explicitly clean up the DBSCAN model object to prevent serialization issues
        del dbscan
        
        # Ensure all data is serializable before returning
        params = ensure_serializable(params)
        
        # Return None for the model object to avoid serialization issues with DBSCAN's internal state
        return params, None

    def validate(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        t7 = Task(
            8 + 4 * (curr_round-1),
            dataflow_tag,
            "ClientValidation",
            dependency=Task(7 + 4 * (curr_round-1), dataflow_tag, "Assemble"),
        )
        t7.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, timestamp]
        t7_input = DataSet("iClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_input)

        # Get validation data
        (x_valid, y_valid, valid_size) = self.valid_data

        # Use global parameters for validation
        if global_param and "core_points" in global_param and len(global_param["core_points"]) > 0:
            core_points = global_param["core_points"]
            core_labels = global_param["core_labels"]
            
            # Convert lists back to numpy arrays if needed
            if isinstance(core_points, list):
                core_points = np.array(core_points)
            if isinstance(core_labels, list):
                core_labels = np.array(core_labels)
            
            # Assign points to nearest core points within eps
            nn = NearestNeighbors(radius=self.eps)
            nn.fit(core_points)
            
            # Find neighbors within eps
            distances, indices = nn.radius_neighbors(x_valid)
            
            # Assign labels based on nearest core points
            y_pred = np.full(len(x_valid), -1)  # Initialize all as noise
            for i, neighbor_idx in enumerate(indices):
                if len(neighbor_idx) > 0:
                    # Assign the label of the nearest core point
                    nearest_idx = neighbor_idx[np.argmin(distances[i])]
                    y_pred[i] = core_labels[nearest_idx]
            
            # Calculate validation metrics
            if len(set(y_pred)) > 1:  # More than one cluster
                silhouette = silhouette_score(x_valid, y_pred)
                calinski = calinski_harabasz_score(x_valid, y_pred)
                self.log_info(fl_ctx, f"Silhouette Score {silhouette:.4f}")
                self.log_info(fl_ctx, f"Calinski-Harabasz Score {calinski:.4f}")
                metrics = {
                    "Silhouette Score": silhouette,
                    "Calinski-Harabasz Score": calinski
                }
            else:
                metrics = {
                    "Silhouette Score": 0.0,
                    "Calinski-Harabasz Score": 0.0
                }
                silhouette = 0.0
        else:
            metrics = {
                "Silhouette Score": 0.0,
                "Calinski-Harabasz Score": 0.0
            }
            silhouette = 0.0

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, silhouette, duration, timestamp]
        
        t7_output = DataSet("oClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_output)
        t7.end()
        
        return metrics, None

    def finalize(self, fl_ctx: FLContext) -> None:
        curr_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        timestamp = datetime.datetime.now()

        t9 = Task(
            9 + 4 * (curr_round),
            dataflow_tag,
            "FinalizeClient",
            dependency=Task(8 + 4 * (curr_round-1), dataflow_tag, "ClientValidation"),
        )
        t9.begin()
        start = perf_counter()
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")

        duration = perf_counter() - start
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t9_output = DataSet("oFinalizeClient", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_output)
        t9.end()