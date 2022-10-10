#!/bin/bash

MAIN_MODULE="/local/hbansal/minidalle"
echo $MAIN_MODULE
cd $MAIN_MODULE

source activate minidalle

python gen_minidalle.py --output_fname minidalle_output_person_s1 --device_id 0 --entity profession --intervention "if all individuals can be [mask] irrespective of their skin color"
python gen_minidalle.py --output_fname minidalle_output_objects_s1 --device_id 0 --entity objects_eis