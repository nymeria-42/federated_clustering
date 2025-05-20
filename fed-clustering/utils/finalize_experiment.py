import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def carregar_info_experimento() -> dict:
    info_path = Path("experiment_info.json")
    if not info_path.exists():
        raise FileNotFoundError(
            "experiment_info.json não encontrado no diretório atual."
        )

    with open(info_path, "r") as f:
        return json.load(f)


def calcular_tempo_decorrido(timestamp_inicio: str) -> str:
    inicio = datetime.strptime(timestamp_inicio, "%Y%m%dT%H%M%S")
    agora = datetime.utcnow()
    delta = agora - inicio
    return str(delta)


def criar_commit_final(duracao: str, hash_valor: str):
    try:
        with open("experiment_duration.txt", "w") as f:
            f.write(f"Duração total: {duracao}\n")

        subprocess.run(["git", "add", "."], check=True)
        msg = f"Finalização do experimento - HASH: {hash_valor} - duração: {duracao}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        print("Commit final criado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando Git: {e}")
        raise


if __name__ == "__main__":
    try:
        info = carregar_info_experimento()
        duracao = calcular_tempo_decorrido(info["timestamp"])
        criar_commit_final(duracao, info["hash_experimento"])
        print(f"Experimento finalizado com duração: {duracao}")
    except Exception as e:
        print(f"Erro ao finalizar experimento: {e}")
