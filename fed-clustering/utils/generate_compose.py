#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate compose.yaml for federated clustering with specified number of clients")
parser.add_argument("--num_clients", "-c", type=int, help="Number of client sites")
parser.add_argument("--output_path", "-o", default="./workspace/fed_clustering/prod_01", help="Output file path for compose.yaml")
parser.add_argument("--data-path", "-d", default=None, help="Path to processed client data (e.g., /absolute/path/to/processed)")

args = parser.parse_args()

num_clients = args.num_clients
output_path = args.output_path
data_path = Path(args.data_path).resolve() if args.data_path else None

# Validate data path if provided
if data_path and not data_path.exists():
    raise FileNotFoundError(f"Data path does not exist: {data_path}")
compose = """services:
  overseer:
    build: nvflare_compose
    command:
      - ${WORKSPACE}/startup/start.sh
    container_name: overseer
    image: ${IMAGE_NAME}
    ports:
      - 8443:8443
    volumes:
      - ./overseer:${WORKSPACE}
  server1:
    build: nvflare_compose
    command:
      - ${PYTHON_EXECUTABLE}
      - -u
      - -m
      - nvflare.private.fed.app.server.server_train
      - -m
      - ${WORKSPACE}
      - -s
      - fed_server.json
      - --set
      - secure_train=true
      - config_folder=config
      - org=nvidia
    container_name: server1
    image: ${IMAGE_NAME}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - DFA_URL=http://host.docker.internal:22000
    ports:
      - 8002:8002
      - 8003:8003
    volumes:
      - ./server1:${WORKSPACE}
      - nvflare_svc_persist:/tmp/nvflare/
"""

# Add client sites
for i in range(1, num_clients + 1):
    # Build volume mounts for this client
    volumes = [
        "      - /tmp/nvflare/dataset:/tmp/nvflare/dataset",
        f"      - ./site-{i}:${{{{WORKSPACE}}}}"
    ]
    
    # If data path is provided, mount the specific client data file
    if data_path:
        client_data_file = data_path / f"client_{i-1}_processed.csv"
        if client_data_file.exists():
            # Mount the file directly into the expected location
            volumes.insert(0, f'      - "{client_data_file}:/tmp/nvflare/dataset/dataset.csv"')
        else:
            print(f"Warning: Data file not found for client {i}: {client_data_file}")
    
    volumes_str = "\n".join(volumes)
    
    site = f"""  site-{i}:
    build: nvflare_compose
    command:
      - ${{PYTHON_EXECUTABLE}}
      - -u
      - -m
      - nvflare.private.fed.app.client.client_train
      - -m
      - ${{WORKSPACE}}
      - -s
      - fed_client.json
      - --set
      - secure_train=true
      - uid=site-{i}
      - org=nvidia
      - config_folder=config
    container_name: site-{i}
    image: ${{IMAGE_NAME}}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - DFA_URL=http://host.docker.internal:22000
    volumes:
{volumes_str}
"""

    compose += site

# Add volumes
compose += """volumes:
  nvflare_svc_persist: null
"""

with open(output_path + "/compose.yaml", 'w') as f:
    f.write(compose)

print(f"compose.yaml generated with {num_clients} clients in {output_path}")
if data_path:
    print(f"Client data mounted from: {data_path}")
else:
    print("WARNING: No data path specified. Use --data-path to mount processed client data.")