#!/bin/bash
# python visualize_attention.py --backbone large --pretrain_backbone omnidata \
# --omnidata_model_path /datasets/home/memmel/PointNav-VO/dpt/pretrained_models \
# --threshold 0.8 --matplotlib_colors True \
# --image_path /datasets/home/memmel/PointNav-VO/imgs_test/img_norm_0_act_1.png

# python visualize_attention.py --backbone base --pretrain_backbone in21k \
# --pretrained_weights /datasets/home/memmel/PointNav-VO/train_log/move_forward/seed_100-vo-noise_1-train-rgb-dd_none_0-m_cen_0-act_-1-model_vo_transformer_act_embed-base-geo_inv_joint_inv_w_1-l_mult_fix_1-1.0_1.0_1.0-dpout_0-e_150-b_20-lr_0.00025-w_de_0.0-20220210_173809360490/checkpoints/ckpt_epoch_54.pth \
# --threshold 0.1 --matplotlib_colors True \
# --image_path /datasets/home/memmel/PointNav-VO/imgs_test/img_norm_0_act_1.png

# python visualize_attention.py --backbone base --pretrain_backbone in21k \
# --pretrained_weights /datasets/home/memmel/PointNav-VO/train_log/move_forward/seed_100-vo-noise_1-train-rgb-dd_none_0-m_cen_0-act_-1-model_vo_transformer_act_embed-base-geo_inv_joint_inv_w_1-l_mult_fix_1-1.0_1.0_1.0-dpout_0-e_150-b_20-lr_0.00025-w_de_0.0-20220210_173809360490/checkpoints/ckpt_epoch_54.pth \
# --threshold 0.5 --matplotlib_colors True \
# --image_path /datasets/home/memmel/PointNav-VO/imgs_test/img_norm_2_act_3.png

# python visualize_attention.py --backbone base --pretrain_backbone in21k \
# --pretrained_weights /datasets/home/memmel/PointNav-VO/train_log/move_forward/seed_100-vo-noise_1-train-rgb-dd_none_0-m_cen_0-act_-1-model_vo_transformer_act_embed-base-geo_inv_joint_inv_w_1-l_mult_fix_1-1.0_1.0_1.0-dpout_0-e_150-b_20-lr_0.00025-w_de_0.0-20220210_173809360490/checkpoints/ckpt_epoch_54.pth \
# --threshold 0.1 --matplotlib_colors True \
# --image_path /datasets/home/memmel/PointNav-VO/imgs_test/img_norm_5_act_3.png

# pip install einops

python visualize_attention.py --backbone base --pretrain_backbone in21k \
--pretrained_weights train_log/vit/vo_vit_base_in21k_act_rgb_depth_dt_20220407_212908153562_s_1/checkpoints/resume_20220412_084122123408/checkpoints/ckpt_epoch_134.pth \
--threshold 0.3 --matplotlib_colors True \
--patch_size 16 --image_size 384 384 \
--backbone base --pretrain_backbone in21k --cls_action True \
--image_path /datasets/home/memmel/PointNav-VO/imgs_test/img_norm_27_act_1.png