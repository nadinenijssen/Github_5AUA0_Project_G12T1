cd src
python test_emb.py mot --exp_id fid_pairloss_m10_lr5_mot17validation11_e10 \
--gpus '0' --reid_dim 128 --arch 'hrnet_18' \
--load_model ../exp/mot/fid_pairloss_m10_lr5_mot17training_e10/model_10.pth \
--test_emb_data "./data/mot17_11.validation" \
--conf_thres 0.35
cd ..

