#!/bin/bash

source $HOME/.env/crit-grad/bin/activate

find -name "offline*" -type d | while read line; do
	if ! grep -Fxq $line synced.txt; then
		wandb sync $line
		echo $line >> synced.txt
	fi
done
deactivate
