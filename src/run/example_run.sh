#!/bin/bash


python -u src/main.py --arch $1 --restype C --blocktype simple --epochs 100 --save-model --results-dir example_run/"$1" --models-dir example_run/"$1" --preds-dir example_run/"$1" --size M --verbose
