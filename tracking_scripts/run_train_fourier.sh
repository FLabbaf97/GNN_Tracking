#!/bin/bash 
source activate paper2022 &&
cd /mlodata1/hokarami/fari/tracking/YeaZ-toolbox && 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && 
pip install torch_geometric && 
pip install -r requirements.txt && 
pip install -e .

cd /mlodata1/hokarami/fari/tracking/tracking_scripts 


# create assgraphs
python -m bread.algo.tracking.build_cellgraphs --config config_fourier_NL_cellgraphs 

# create assgraphs
python -m bread.algo.tracking.build_assgraphs --config config_fourier_NL_assgraphs 
python -m bread.algo.tracking.build_assgraphs --config config_fourier_NL_assgraphs2


# train
python train.py --config config_fourier_NL_assgraphs 
python train.py --config config_fourier_NL_assgraphs 
python train.py --config config_fourier_NL_assgraphs 
# python train.py --config config_fourier_NL_assgraphs &&
# python train.py --config config_fourier_NL_assgraphs &&

# test all
# python test_p.py --config config_fourier_NL_test --model '/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_f10_locality_False/_100KB_/2023-12-19_12:47:30'
