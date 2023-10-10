#!bash

# This script is used to train the model.

conda activate VisIrNet
# python Train.py --config-file skydata_default_config.json
python Train.py --config-file vedai_default_config.json
