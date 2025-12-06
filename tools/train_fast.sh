export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
storyteller-train --config configs/fast_train.yaml
