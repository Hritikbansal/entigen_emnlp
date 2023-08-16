#!/bin/bash

MAIN_MODULE="/u/home/h/hbansal/stable_diffusion"
echo $MAIN_MODULE
cd $MAIN_MODULE

# python inference_sd.py --entity geographical --output_fname sd_profession_c3 --intervention "from different cultures"
# python inference_sd.py --entity geographical --output_fname sd_profession_c4 --intervention "if compassion is the basis of morality"
# python inference_sd.py --entity geographical --output_fname sd_profession_c5 --intervention "if honesty is the best policy"

python inference_sd.py --entity cultural --output_fname sd_profession_c3 --intervention "from different cultures"
python inference_sd.py --entity cultural --output_fname sd_profession_c4 --intervention "if compassion is the basis of morality"
python inference_sd.py --entity cultural --output_fname sd_profession_c5 --intervention "if honesty is the best policy"



python inference_sd.py --entity objects --output_fname sd_objects_o1 
python inference_sd.py --entity objects_ei --output_fname sd_objects_o2
python inference_sd.py --entity objects_eir --output_fname sd_objects_o3
python inference_sd.py --entity objects --output_fname sd_objects_o4 --intervention "if compassion is the basis of morality"