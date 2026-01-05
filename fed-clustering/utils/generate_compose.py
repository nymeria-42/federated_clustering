#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description="Generate compose.yaml for federated clustering with specified number of clients")
parser.add_argument("--num_clients", "-c", type=int, help="Number of client sites")
parser.add_argument("--output_path", "-o",default = "./workspace/fed_clustering/prod_01", help="Output file path for compose.yaml")

args = parser.parse_args()

num_clients = args.num_clients
output_path = args.output_path

# Base compose content
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
      - /tmp/nvflare/dataset:/tmp/nvflare/dataset
      - ./site-{i}:${{WORKSPACE}}
"""

    compose += site

# Add volumes
compose += """volumes:
  nvflare_svc_persist: null
"""

with open(output_path + "/compose.yaml", 'w') as f:
    f.write(compose)

print(f"compose.yaml generated with {num_clients} clients in {output_path}")