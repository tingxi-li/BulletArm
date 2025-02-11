#!/bin/bash

for ((i=0;i<86;i++));do

{
    python bulletarm_baselines/fc_dqn/scripts/collect_data.py --device_name=cuda --num_process=0 --load_model_pre="/home/weilab/tingxi/snapshot" --num_object=1 --object_index=${i}
    
}
done