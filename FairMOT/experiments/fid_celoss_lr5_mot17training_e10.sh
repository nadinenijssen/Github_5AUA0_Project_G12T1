cd src
python train.py mot --exp_id fid_celoss_lr5_mot17training_e10 \
--gpus '0' --num_epochs 10 --lr 1e-5 --batch_size 8 --reid_dim 128 \
--arch 'hrnet_18' --load_model '../models/model_45.pth' --freeze backbone_det \
--train_data "./data/mot17.training" --data_dir '/workspace/datasets/'
cd ..