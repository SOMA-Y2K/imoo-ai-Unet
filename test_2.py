import os
import subprocess

# Set LOCAL_RANK environment variable if needed
os.environ['LOCAL_RANK'] = '0'  # You can set this to the appropriate value

command = [
    "torchrun",
    "--nproc_per_node=1",  # You can adjust this as needed
    "--master_port=4736",
    "test_tryon.py",
    "--name", "test_gpvtongen_vitonhd_unpaired_1109",
    "--resize_or_crop", "None", "--verbose", "--tf_log",
    "--batchSize", "10", "--num_gpus", "1", "--label_nc", "14", "--launcher", "pytorch",
    "--PBAFN_gen_checkpoint", "checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth",
    "--dataroot", "VITON-HD",
    "--image_pairs_txt", "test_pairs_unpaired_1018.txt",
    "--warproot", "sample/test_partflow_vitonhd_unpaired_1109"
]

subprocess.run(command)

