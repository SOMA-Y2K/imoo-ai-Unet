import os
import subprocess


# Set LOCAL_RANK environment variable if needed
os.environ['LOCAL_RANK'] = '0'  # You can set this to the appropriate value

command = [
    "torchrun", "test_warping.py",
    "--name", "test_partflow_vitonhd_unpaired_1109",
    "--PBAFN_warp_checkpoint", "checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth",
    "--resize_or_crop", "None", "--verbose", "--tf_log",
    "--batchSize", "2", "--num_gpus", "8", "--label_nc", "14", "--launcher", "pytorch",
    "--dataroot", "VITON-HD",
    "--image_pairs_txt", "test_pairs_unpaired_1018.txt"
]

subprocess.run(command)



