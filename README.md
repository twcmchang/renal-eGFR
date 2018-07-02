# Kidney eGFR

1. ALL ipynb files in /home/timliu/kidney

2. Current model: /data/put_data/timliu/kidney/model_for_prediction/0627

3. data:
	- new data 2016: 
		- IMAGE:/data/put_data/timliu/kidney_2016/png_ge_crop_filter/
		- TABLE: /data/put_data/timliu/kidney_2016/meta_final.csv

	- old data: /data/put_data/timliu/kidney/pro_img/GE/crop_img/
		- integrated into meta_final.csv

4. ipynb files:
	- table_merge.ipynb: preprocess

	- select_ge_2016.ipynb: preprocess

	- mapping_2016.ipynb: preprocess

	- data_preprocess.ipynb:
		- cut off fan-shaped region

	- polar_coordinate_transform.ipynb
		- crop, polar transformation

	- kidney_pypy_sig_pretrain_mod_noscale-5fold-Copy1.ipynb

	- model_ensemble.ipynb
		- run every model

	- tree_based_prediction_cleared-xxx.ipynb
		- XGBoost

	- boostrapping_confidence_interval.ipynb
		- run boostrapping for confidence interval (on paper)
		- obtain a summary_df
