cd src
python train.py mot --exp_id fid_pairloss_m10_lr5_mot17training_e30 \
--gpus '3' --num_epochs 30 --lr 1e-5  --lr_step '20,25'  --batch_size 8 --reid_dim 128 \
--arch 'hrnet_18' --load_model '../models/model_45.pth' --freeze backbone_det \
--train_data "./data/mot17.training" --data_dir '/workspace/datasets/' \
--id_loss 'pairwise' --pairwise_margin 10.0 --pairwise_sampling 'hardest' --positives_sampling True
cd ..