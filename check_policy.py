import torch
checkpoint = torch.load('/datasets/home/memmel/PointNav-VO/pretrained_ckpts/pointnav2021_partsey/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31_ckpt.94_spl_0.8003.pth')
print(checkpoint["config"])

checkpoint = torch.load('/datasets/home/memmel/PointNav-VO/pretrained_ckpts/rl/no_tune/rl_no_tune.pth')
print(checkpoint['config'])