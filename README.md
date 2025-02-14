# Federated k-Means Clustering with Scikit-learn

## Overview
This project demonstrates Federated Learning (FL) with k-Means clustering using Scikit-learn. The aggregation follows the MiniBatch k-Means approach, where clients perform local training, and a central server aggregates their results to update global cluster centers.

### Federated k-Means Process
Each FL round consists of:
1. **Local Training**: Clients initialize with global centers and train MiniBatchKMeans on local data.
2. **Global Aggregation**: The server collects cluster centers and counts from all clients, updates the global model, and redistributes it for the next round.

#### Initialization Strategy
- Clients use k-means++ for initial cluster centers.
- The server aggregates initial centers using a round of k-means to determine the global starting point.

---

## Running Federated Learning with Proper Data Splitting

### Data Partitioning
For an experiment with `K` clients, the dataset is split into `K+1` non-overlapping parts:
- `K` parts for client training.
- `1` common validation dataset.

### Simulating Data Imbalance
Clients' data sizes can be distributed using different strategies:
- **Uniform**: Equal data for all clients.
- **Linear**: Data size grows linearly with client ID.
- **Square**: Data size increases quadratically.
- **Exponential**: Best suited for small `K` (e.g., `K=5`) to simulate extreme imbalance.

---

## Capturing Provenance with DfAnalyzer

DfAnalyzer is a library designed to capture provenance data, which includes:
- **Prospective Provenance**: The "recipe" of the experiment, capturing configurations before execution.
- **Retrospective Provenance**: Logs and tracks execution results, including data transformations and FL training steps.

### Setting Up DfAnalyzer
1. Load the DfAnalyzer Docker image:
   ```bash
   docker load < dfanalyzer_image.tar.gz
   ```
2. Deploy the DfAnalyzer container:
   ```bash
   docker compose up dfanalyzer
   ```
3. Ensure DfAnalyzer is running in the background before starting experiments.

---

## NVFlare Provisioning
NVFlare is used to set up the federated learning infrastructure.

### Provisioning
Run:
```bash
nvflare provision
```
This creates a `workspace/fed_clustering` directory with the following structure:
```
workspace
└── fed_clustering
    ├── prod_00
    │   ├── admin@nvidia.com
    │   ├── server1
    │   ├── site-1
    │   ├── site-2
    │   ├── overseer
    │   └── compose.yaml
    └── resources
```

### Running NVFlare
Navigate to `prod_00` and launch FL components:
```bash
docker compose up
```

---

## Running the Experiment

### 1. Preparing Data and Configuration
```bash
./prepare_data.sh
./prepare_job_config.sh
```
Manually copy `sklearn_2_uniform` to `workspace/fed_clustering/prod_00/admin@nvidia.com/startup/transfer/`.

### 2. Distributing Data to Clients
Create dataset folders inside the containers:
```bash
docker exec -it site-1 mkdir -p /tmp/nvflare/dataset
```
Copy data to each client:
```bash
docker cp /tmp/nvflare/dataset/des.csv site-1:/tmp/nvflare/dataset
```
Repeat for all sites.

### 3. Running the FL Server
Inside `prod_00`, start the FL admin panel:
```bash
./admin@nvidia.com/startup/fl_admin.sh
```
Log in with:
```
admin@nvidia.com
```

### 4. Configuring the Hosts File
1. Get the local hostname:
   ```bash
   hostname -I | awk '{print $1}'
   ```
2. Add it to `/etc/hosts`:
   ```bash
   sudo vim /etc/hosts
   ```
   Add:
   ```
   {IP}  server server1 overseer
   ```

### 5. Checking FL System Status
```bash
check_status [server|client]
```

### 6. Submitting and Running the Experiment
```bash
submit_job sklearn_kmeans_2_uniform
```
Monitor DfAnalyzer for provenance tracking.

---

## Conclusion
This project demonstrates federated k-Means clustering using NVFlare, Scikit-learn, and DfAnalyzer. Provenance data is captured throughout, ensuring transparency and reproducibility of FL experiments.

