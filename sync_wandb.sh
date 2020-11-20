#!/bin/bash

source $HOME/.env/crit-grad/bin/activate

find -name "offline*" -type d | while read line; do
    wandb sync $line
done

deactivate