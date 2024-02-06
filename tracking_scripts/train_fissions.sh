#!/bin/bash 
# source activate paper2022 &&
# cd /mlodata1/hokarami/fari/tracking/YeaZ-toolbox && 
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && 
# pip install torch_geometric && 
# pip install -r requirements.txt && 
# pip install -e .

# cd /mlodata1/hokarami/fari/tracking/tracking_scripts 


# create cellgraphs one by one
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_01_segmentation.h5" --nn_threshold 60 --frame_max 144 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_02_segmentation.h5" --nn_threshold 60 --frame_max 144 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_01_segmentation.h5" --nn_threshold 60 --frame_max 74 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_03_segmentation.h5" --nn_threshold 60 --frame_max 114 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_05_segmentation.h5" --nn_threshold 60 --frame_max 114 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_07_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_fourier_features"
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_15_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_fourier_features"
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_20_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_fourier_features"
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_fourier_features_location" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_30_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_fourier_features"


# create assgraphs
# python -m bread.algo.tracking.build_assgraphs --config config_fission_fourier_features_location --cellgraphs_dir "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_fourier_features/wt_pom1D_01_15_R3D_REF_dv_trk_segmentation"

# python -m bread.algo.tracking.build_assgraphs --config config_fission_fourier_features_location 



# # test all
# python test_single_model.py --config config_fission_fourier_location --model '/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_f10_locality_False/_100KB_/2023-12-19_12:47:30' --resultdir "/mlodata1/hokarami/fari/tracking/results" --algo "hungarian" --data "fission_all" --assgraphs "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/fissions/ass_graphs_fourier_10_f10_locality_False"


# Do the same for other config
# create cellgraphs one by one
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_01_segmentation.h5" --nn_threshold 60 --frame_max 144 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_02_segmentation.h5" --nn_threshold 60 --frame_max 144 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_01_segmentation.h5" --nn_threshold 60 --frame_max 74 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_03_segmentation.h5" --nn_threshold 60 --frame_max 114 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_05_segmentation.h5" --nn_threshold 60 --frame_max 114 --frame_min 0
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_07_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features"
python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_15_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features"
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_20_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features"
# python -m bread.algo.tracking.build_cellgraphs --config "config_fission_features_coordinate" --input_segmentation "/mlodata1/hokarami/fari/bread/data/fission/wt_masks/wt_pom1D_01_30_R3D_REF_dv_trk_segmentation.h5" --nn_threshold 60 --frame_max 179 --frame_min 0 --output_folder "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features"


# create assgraphs
python -m bread.algo.tracking.build_assgraphs --config config_fission_features_coordinate --cellgraphs_dir "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features/wt_pom1D_01_15_R3D_REF_dv_trk_segmentation"

# python -m bread.algo.tracking.build_assgraphs --config config_fission_features_coordinate 


# train for features
# python train.py --config config_fission_features_coordinate

# train for fourier and features
# python train.py --config config_fission_fourier_features_location
# python train.py --config config_fission_fourier_location
# python train.py --config config_fission_fourier_location
