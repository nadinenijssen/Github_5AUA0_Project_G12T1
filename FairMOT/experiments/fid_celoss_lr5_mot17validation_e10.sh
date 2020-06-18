cd src
python track.py mot --exp_id fid_celoss_lr5_mot17validation_e10 \
--gpus '0' --reid_dim 128 --arch 'hrnet_18' \
--load_model ../exp/mot/fid_celoss_lr5_mot17training_e10/model_10.pth \
--validation_mot17 True \
--distance_metric 'cosine' --conf_thres 0.35
cd ..