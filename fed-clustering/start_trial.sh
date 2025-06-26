#!/bin/bash

# Executa o script Python e captura o caminho da worktree como última linha
WORKTREE_PATH=$(python3 utils/start_trial.py | tail -n 1)

# Muda para o diretório e abre shell interativo
cd "$WORKTREE_PATH" && exec $SHELL