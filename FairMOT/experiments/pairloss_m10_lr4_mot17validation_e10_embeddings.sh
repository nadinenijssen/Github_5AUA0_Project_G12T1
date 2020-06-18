cd src
python test_emb.py mot --exp_id pairloss_m10_lr4_mot17validation_e10 \
--gpus '0' --reid_dim 128 --arch 'hrnet_18' \
--load_model ../exp/mot/pairloss_m10_lr4_mot17training_e10/model_10.pth \
--test_emb_data "./data/mot17.validation" \
--conf_thres 0.35
cd ..
