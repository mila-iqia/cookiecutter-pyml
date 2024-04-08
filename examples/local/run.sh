set -e
amlrt-train --data ../data --output output --config ../config.yaml config.yaml --start-from-scratch
amlrt-eval --data ../data --config ../config.yaml config.yaml --ckpt-path output/best_model/model.ckpt
