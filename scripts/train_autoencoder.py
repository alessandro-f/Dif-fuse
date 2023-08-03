import sys
# put your path here
sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
from utils.arg_parsing import parse_args
import pprint
import torch.nn as nn
from utils.storage import (
    build_experiment_folder,
    save_checkpoint,
    restore_model,
)
import random
import glob
import tarfile
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from guided_diffusion.brain_datasets import *
from utils.metric_tracking import (
    MetricTracker,
    compute_accuracy,
)
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
from autoencoder_architectures import  AE_no_bottleneck_6_12_16
import torch
from torch.utils.data import DataLoader
import tqdm

main_start = datetime.now()
dt_string = main_start.strftime("%d/%m/%Y %H:%M:%S")
print("Start main() date and time =", dt_string)
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'



torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)


device = (
    torch.cuda.current_device()
    if torch.cuda.is_available() and args.num_gpus_to_use > 0
    else "cpu"
)
print(
    "Device: {} num_gpus: {}  torch.cuda.is_available() {}".format(
        device, args.num_gpus_to_use, torch.cuda.is_available()
    )
)
args.device = device

height = 256
width = 256
channels = 4
args.num_workers = 4

train_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_train.csv',
    transform=None,
    only_positive=False,
    only_negative=True,
    only_flair=False)

val_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_val.csv',
    transform=None,
    only_positive=False,
    only_negative=True,
    only_flair=False)

test_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_test.csv',
    transform=None,
    only_positive=False,
    only_negative=True,
    only_flair=False)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# Save a snapshot of the current state of the code.

saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
    experiment_name=args.experiment_name,
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),
)

snapshot_code_folder = saved_models_filepath
if not os.path.exists(snapshot_code_folder):
    os.makedirs(snapshot_code_folder)
snapshot_filename = "{}/snapshot.tar.gz".format(snapshot_code_folder)
filetypes_to_include = [".py"]
all_files = []
for filetype in filetypes_to_include:
    all_files += glob.glob("**/*.py", recursive=True)
with tarfile.open(snapshot_filename, "w:gz") as tar:
    for file in all_files:
        tar.add(file)
print(
    "saved_models_filepath: {} logs_filepath: {} images_filepath: {}".format(
        saved_models_filepath, logs_filepath, images_filepath
    )
)

print("-----------------------------------")
pprint.pprint(args, indent=4)
print("-----------------------------------")


mod = 'AE_brats_no_bottleneck_6_12_16_2d_256x256_MSE'
enc_out = 512


ae = AE_no_bottleneck_6_12_16(batch_size= args.batch_size, input_height = 256, enc_type='my_resnet18', first_conv=False, maxpool1=False, enc_out_dim=enc_out, latent_dim=int(enc_out/2), lr=0.0001)

ae = ae.to(device)

optimizer = optim.Adam(ae.parameters(), lr=3e-4)

criterion = torch.nn.MSELoss()



if args.scheduler == "CosineAnnealing":
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epochs, eta_min=args.lr_min
    )
else:
    scheduler = MultiStepLR(optimizer, milestones=int(args.max_epochs / 10), gamma=0.1)

############################################################################# Restoring

restore_fields = {
    "model": ae,
    "optimizer": optimizer,
    "scheduler": scheduler,
}
#####################################
# log the logits, loss etc.
log_dict = {}
log_dict["logits"] = {}
log_dict["loss"] = {}
log_dict["MTL_loss"] = []
log_dict["gradients"] = {}

for name, param in ae.named_parameters():
    log_dict["gradients"][name] = []

#######################################
start_epoch = 0


################################################################################# Metric
# track for each iteration with input of raw output from NN and targets
metrics_to_track = {
    "mse": lambda x, y: torch.nn.MSELoss()(x, y).item(),
    "accuracy": compute_accuracy,
}


############################################################################## Training

def train_iter( model, x, ids, iteration, epoch, set_name, train_loss):

    x = x.to(device)
    model = model.train()
    z = model.module.encoder(x)
    x_hat = model.module.decoder(z)

    loss = criterion(x_hat, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        loss.item()
    )
    train_loss+=loss.item()

    return log_string, train_loss


def eval_iter(model, x, ids, iteration, epoch, set_name, eval_loss):

    x = x.to(device)

    with torch.no_grad():
        model = model.eval()
        z = model.module.encoder(x)
        x_hat = model.module.decoder(z)
        loss = criterion(x_hat, x)


    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        loss.item()
    )

    eval_loss += loss.item()

    return log_string, eval_loss


def run_epoch(epoch, model, training, data_loader, metric_tracker):
    iterations = epoch * (len(data_loader) )
    print(f"{len(data_loader) } batches.")

    train_loss = 0
    eval_loss = 0
    with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:
        # Clear this for every epoch. Only save the best epoch results.
        for x, _, _,ids in data_loader:

            if training:
                log_string, train_loss = train_iter(
                    model=model,
                    x=x,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    train_loss = train_loss
                )

            else:
                log_string, eval_loss = eval_iter(
                    model=model,
                    x=x,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    eval_loss = eval_loss
                )

            pbar.set_description(log_string)
            pbar.update(1)
            iterations += 1
    return eval_loss if not training else train_loss


def save_model(saved_models_filepath, latest_epoch_filename, best_epoch_filename, is_best):
    if args.save:
        state = {
            "args": args,
            "epoch": epoch,
            "model": ae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        # save the latest epoch model
        epoch_pbar.set_description(
            "Saving latest checkpoint at {}/{}".format(
                saved_models_filepath, latest_epoch_filename
            )
        )
        save_checkpoint(
            state=state,
            directory=saved_models_filepath,
            filename=latest_epoch_filename,
            is_best=False,
        )
        # save the best model.
        best_model_path = ""
        if is_best:
            best_model_path = save_checkpoint(
                state=state,
                directory=saved_models_filepath,
                filename=best_epoch_filename,
                is_best=True,
            )
            epoch_pbar.set_description(
                "Saving best checkpoint at {}/{}".format(
                    saved_models_filepath, best_model_path
                )
            )
    return best_model_path


if __name__ == "__main__":
    loss_weight_dict = None


    if args.resume:
        start_epoch = restore_model(restore_fields, path=saved_models_filepath, device=device)

    if args.num_gpus_to_use > 1:
        ae = nn.DataParallel(ae)
        ae.module.encoder = nn.DataParallel(ae.module.encoder)
        ae.module.decoder = nn.DataParallel(ae.module.decoder)

    metric_tracker_train = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="training"
    )
    metric_tracker_val = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="validation"
    )

    metric_tracker_test = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="testing"
    )

    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        best_loss = 2.0
        best_epoch = 0
        each_epoch_acc = []


        for epoch in range(start_epoch, args.max_epochs):
            print()
            print(f"Running epoch {epoch}")
            best_epoch_stats = open(logs_filepath + "/best_epoch_stats.txt", "w")
            tot_train_loss = run_epoch(
                epoch,
                data_loader=train_loader,
                model=ae,
                training=True,
                metric_tracker=metric_tracker_train,
            )
            tot_eval_loss = run_epoch(
                epoch,
                data_loader=val_loader,
                model=ae,
                training=False,
                metric_tracker=metric_tracker_val,
            )
            #     # run_epoch(
            #     #     epoch,  # only test after all the trainning epochs
            #     #     model=model,
            #     #     training=False,
            #     #     data_loader=test_set_loader,
            #     #     metric_tracker=metric_tracker_test,
            #     # )
            #
            scheduler.step()

            is_best= False




            current_val_loss = tot_eval_loss

            print(f"\ncurrent_train_loss: {tot_train_loss:.4f}")
            print(f"\ncurrent_val_loss: {current_val_loss:.4f}")
            latest_epoch_filename = "latest_ckpt.pth.tar"
            best_epoch_filename = "ckpt.pth.tar"
            if current_val_loss <= best_loss:
                best_loss = current_val_loss
                is_best = True
                best_epoch = epoch
                best_model_save_path = save_model(saved_models_filepath=saved_models_filepath,
                                                  latest_epoch_filename=latest_epoch_filename,
                                                  best_epoch_filename=best_epoch_filename,
                                                  is_best=is_best,
                                                  )
            else:
                is_best = False
                save_model(saved_models_filepath=saved_models_filepath,
                           latest_epoch_filename=latest_epoch_filename,
                           best_epoch_filename=best_epoch_filename,
                           is_best=is_best,
                           )
            print(
                "\nAll tasks till epoch: {} best_loss: {:.4f} best_epoch: {:}".format(
                    epoch, best_loss, best_epoch
                )
            )
            epoch_pbar.set_description("")
            epoch_pbar.update(1)
            # Save important stats snapshot.
            # Track if best epoch changed, only compute best epoch stats if changed
            print()
            best_epoch_stats.write(f"Till epoch: {epoch}\n")
            print(f"Best_val_epoch: {best_epoch}")
            print(f"Best_val_loss: {best_loss:.4f}")
            best_epoch_stats.write(f"Best_val_epoch: {best_epoch}\n")
            best_epoch_stats.write(f"Best_val_loss: {best_loss:.4f}\n")


            each_epoch_acc.append(current_val_loss)




    main_end = datetime.now()
    dt_string = main_end.strftime("%d/%m/%Y %H:%M:%S")
    print("End main() date and time =", dt_string)

    main_execute_time = main_end - main_start
    print("main() execute time: {}".format(main_execute_time))
    sys.exit(0)

