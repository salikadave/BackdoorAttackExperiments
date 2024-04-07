#!/bin/bash

# 1. Pass absolute path
# 2. Pass env_name

if [ $# -lt 2 ]; then
    echo "Usage: <absolute_path> <env_name>"
    exit 1
fi

# Change directory one level up
cd $1

# source env/Scripts/activate

activate_env(){
    source env/Scripts/activate
}

deactivate_env(){
    . ${VENV}/deactivate
}
# [-r "deactivate"]
activate_env
pip -V