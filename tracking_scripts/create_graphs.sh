# # make cell_graphs for all images
# python -m bread.algo.tracking.build_cellgraphs --config config_fourier_and_f10_NL_5colonies &
# python -m bread.algo.tracking.build_cellgraphs --config config_fourier_and_f10_NL_benchmarks &
# python -m bread.algo.tracking.build_cellgraphs --config config_fourier_and_f10_NL_cellgraphs 

# make ass_graphs for all images

python -m bread.algo.tracking.build_assgraphs --config config_fourier_and_f10_NL_assgraphs_5colonies
python -m bread.algo.tracking.build_assgraphs --config config_fourier_and_f10_NL_assgraphs_benchmarks 
python -m bread.algo.tracking.build_assgraphs --config config_fourier_and_f10_NL_assgraphs
# Train GNN

# test