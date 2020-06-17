cd src
python train.py mot --exp_id all_hrnet --gpus 3 --batch_size 4 --reid_dim 128 --arch 'hrnet_18' \
--data_dir "/workspace/datasets/" --load_model '../models/hrnet_10_epoch.pth' --lr_step '55,65' --num_epochs 75 \
--freeze backbone
cd .