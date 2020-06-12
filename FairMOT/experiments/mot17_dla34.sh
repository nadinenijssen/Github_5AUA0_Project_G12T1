cd src
python train.py mot --exp_id all_hrnet --gpus 2,3 --batch_size 4 --reid_dim 128 --load_model '../models/ctdet_coco_dla_2x.pth' --data_dir "/workspace/datasets/"
cd .