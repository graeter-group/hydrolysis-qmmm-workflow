#!/bin/env bash

source assets/_modules.sh

UV_PROJECT_ENVIRONMENT=".venv"

if [[ "$(hostname)" == "cascade"* ]]; then
  ## cluster
  echo "cascade"
  UV_PROJECT_ENVIRONMENT=".venv-cascade"
elif [[ "$(hostname)" == "pop-desktop" ]]; then
  echo "local ws"
elif [[ "$(hostname)" == "pop-laptop" ]]; then
  # laptop
  :
elif [[ "$(hostname)" == "pop-os" ]]; then
  # laptop
  :
else
  ## workstation
  # source /sw/mbm/buhrjk/venvs/qmmm/bin/activate
  UV_PROJECT_ENVIRONMENT="/data/sw/venv"
fi

if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
  source $UV_PROJECT_ENVIRONMENT/bin/activate
fi

export UV_PROJECT_ENVIRONMENT
