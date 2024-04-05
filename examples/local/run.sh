set -e
amlrt-train --data ../data --output output --configs ../config.yaml config.yaml --start-from-scratch
amlrt-eval --data ../data --configs ../config.yaml config.yaml --ckpt-path output/best_model/model.ckpt
