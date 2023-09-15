import os
import subprocess

# Set LOCAL_RANK environment variable if needed
os.environ['LOCAL_RANK'] = '0'  # You can set this to the appropriate value

command = [
    "torchrun",  # Use torchrun instead of torch.distributed.launch
    "--master_port=4736",
    "train_tryon.py",
    "--name", "gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029",
    "--resize_or_crop", "None", "--verbose", "--tf_log",
    "--dataset", "vitonhd", "--resolution", "512",
    "--batchSize", "10", "--num_gpus", "8", "--label_nc", "14", "--launcher", "pytorch",
    "--dataroot", "VITON-HD",
    "--image_pairs_txt", "train_pairs_1018.txt",
    "--warproot", "sample/test_gpvton_lrarms_for_training_1029",
    "--display_freq", "50", "--print_freq", "25", "--save_epoch_freq", "10", "--write_loss_frep", "25",
    "--niter_decay", "0", "--niter", "200",
    "--lr", "0.0005"
]

subprocess.run(command)

