#!/bin/bash 
source activate paper2022 &&
cd /mlodata1/hokarami/fari/tracking/YeaZ-toolbox && 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && 
pip install torch_geometric && 
pip install -r requirements.txt && 
pip install -e .

cd /mlodata1/hokarami/fari/tracking/tracking_scripts 


# create assgraphs
# python -m bread.algo.tracking.build_cellgraphs --config config_fourier_and_f10_NL_cellgraphs 

# create assgraphs
# python -m bread.algo.tracking.build_assgraphs --config config_fourier_and_f10_NL_cellgraphs 

# train
# python train.py --config config_fourier_and_f10_NL_test &&
# python train.py --config config_fourier_and_f10_NL_assgraphs &&
# python train.py --config config_fourier_and_f10_NL_assgraphs &&
# python train.py --config config_fourier_and_f10_NL_assgraphs &&
# python train.py --config config_fourier_and_f10_NL_assgraphs &&

# test all
# python test_models.py --data 'colony_0567_test_set_1234567_dt_1234_t_all' --resultdir "/mlodata1/hokarami/fari/tracking/results" --algo "hungarian",
python test_models.py --data 'colony_056_dt_1234_t_all' --resultdir "/mlodata1/hokarami/fari/tracking/results" --algo "hungarian",
python test_models.py --data 'colony_7_test_set_1234567_dt_1_t_all' --resultdir "/mlodata1/hokarami/fari/tracking/results" --algo "hungarian",
# python test_models.py --data 'colony_5__dt_1_t_all' --resultdir "/mlodata1/hokarami/fari/tracking/results" --algo "hungarian",
