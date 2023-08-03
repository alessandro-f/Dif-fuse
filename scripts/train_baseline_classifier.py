import sys
# put your path here
sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
from utils.arg_parsing import parse_args
import os
import pprint
import numpy as np
import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

main_start = datetime.now()
dt_string = main_start.strftime("%d/%m/%Y %H:%M:%S")
print("Start main() date and time =", dt_string)

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'

from utils.storage import (
    build_experiment_folder,
    save_checkpoint,
    restore_model,
    restore_model_from_path, save_metrics_dict_in_pt,
)
from guided_diffusion.brain_datasets import *
import random
import glob
import tarfile
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.metric_tracking import (
    MetricTracker,
    compute_accuracy,

)
from sklearn.metrics import roc_auc_score
import torch



args = parse_args()

task_dict = {'healthy_no_healthy': "healthy"}

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else "cpu"
)

train_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_train.csv',
    transform=None,
    only_positive=False,
    only_negative=False,
    only_flair=False)

val_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_val.csv',
    transform=None,
    only_positive=False,
    only_negative=False,
    only_flair=False)

test_dataset = BRATSDataset(
    dataset_root_folder_filepath='data/brats2021_slices/images',
    df_path='data/brats2021_test.csv',
    transform=None,
    only_positive=False,
    only_negative=False,
    only_flair=False)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)


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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

print("-----------------------------------")
pprint.pprint(args, indent=4)
print("-----------------------------------")


model = torchvision.models.resnet50(progress=False)
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
model.fc = nn.Linear(2048, 2)
best_model = torchvision.models.resnet50(progress=False)
best_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
best_model.fc = nn.Linear(2048, 2)
# print(model)

model = model.to(device)
best_model = best_model.to(device)

if args.num_gpus_to_use > 1:
    model = nn.DataParallel(model)
    best_model = nn.DataParallel(best_model)

zero = 52051
one = 79113

weights = [1-(zero/(zero+one)), zero/(zero+one)]
print('weights', weights)

criterion= nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device))


if args.optim.lower() == "sgd":
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
else:
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

if args.scheduler == "CosineAnnealing":
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=args.max_epochs, eta_min=args.lr_min
    )
else:
    scheduler = MultiStepLR(optimizer, milestones=int(args.max_epochs / 10), gamma=0.1)

############################################################################# Restoring

restore_fields = {
    "model": model,
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
for task_name in task_dict.keys():
    log_dict["logits"][task_name] = torch.Tensor()
    log_dict["loss"][task_name] = []
for name, param in model.named_parameters():
    log_dict["gradients"][name] = []

#######################################
start_epoch = 0


def resume_from_latest_checkpoint(resume=args.resume, saved_models_filepath=None):
    if resume:
        print("-----------Resume------------")
        print("Before resuming, model parameters")
        for name, param in model.named_parameters():
            print(name, param.shape)

        resume_epoch = restore_model(
            restore_fields, path=saved_models_filepath, device=device
        )
        if resume_epoch == -1:
            print("Failed to load from {}/ckpt.pth.tar".format(saved_models_filepath))
        else:
            start_epoch = resume_epoch + 1
        print("resume from start_epoch: {}".format(start_epoch))

        for name, param in model.named_parameters():
            print(name, param.shape)
        print()


################################################################################# Metric
# track for each iteration with input of raw output from NN and targets
metrics_to_track = {
    "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
    "accuracy": compute_accuracy,
}


############################################################################## Training
def save_task_stats(model, task_dict, metric_tracker, logits_dict, loss_dict=None, MTL_loss=None):
    for task_name in task_dict.keys():
        if len(metric_tracker.log_dict["logits"][task_name]) == 0:
            metric_tracker.log_dict["logits"][task_name] = logits_dict[task_name].detach().cpu()
        else:
            metric_tracker.log_dict["logits"][task_name] = \
                torch.cat((metric_tracker.log_dict["logits"][task_name],
                           logits_dict[task_name].detach().cpu()), 0)
        if loss_dict is not None:
            metric_tracker.log_dict["loss"][task_name].append(loss_dict[task_name].detach().cpu())
    if MTL_loss is not None:
        metric_tracker.log_dict["MTL_loss"].append(MTL_loss.detach().cpu())
    for name, param in model.named_parameters():
        if param.grad is not None:
            metric_tracker.log_dict["gradients"][name].append(torch.mean(torch.abs(param.grad)).cpu())


def train_iter(metric_tracker, model, x,y, ids, iteration, epoch, set_name):


    inputs = x.to(device)
    targets = torch.tensor(y, dtype=torch.long).to(device)

    model = model.train()
    logits = model(inputs)
    logits_dict = {}
    targets_dict = {}
    logits_dict['healthy_no_healthy'] = logits
    targets_dict['healthy_no_healthy'] = targets

    loss = criterion(
        input=logits,
        target=targets)

    metric_tracker.push(epoch, iteration, logits_dict, targets_dict)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        "".join(
            "{}: {:0.4f}; ".format(key, value[-1])
            if key not in ["epochs", "iterations"] and len(value) > 0
            else ""
            for key, value in metric_tracker.metrics.items()
        ),
    )

    return log_string


def eval_iter(metric_tracker, model, x, y, ids, iteration, epoch, set_name, final_pred,final_prob, final_label):

    inputs = x.to(device)
    targets = torch.tensor(y, dtype=torch.long).to(device)


    with torch.no_grad():
        model = model.eval()
        logits = model(inputs)
        # after removing the inputs which we don't back prop

    logits_dict = {}
    targets_dict = {}
    logits_dict['healthy_no_healthy'] = logits
    targets_dict['healthy_no_healthy'] = targets


    metric_tracker.push(epoch, iteration, logits_dict, targets_dict)
    m = nn.Softmax(dim=-1)

    for i, el in enumerate(torch.argmax(m(logits_dict['healthy_no_healthy']), dim=1)):
            final_pred.append(int(torch.argmax(m(logits_dict['healthy_no_healthy']), dim=1)[i]))
            final_prob.append(m(logits_dict['healthy_no_healthy'])[:, 1][i].item())


    for i, el in enumerate(targets_dict['healthy_no_healthy']):
            final_label.append(int(targets_dict['healthy_no_healthy'][i]))


    log_string = "{}, epoch: {} {} iteration: {}; {}".format(
        args.experiment_name,
        epoch,
        set_name,
        iteration,
        "".join(
            "{}: {:0.4f}; ".format(key, value[-1])
            if key not in ["epochs", "iterations"] and len(value) > 0
            else ""
            for key, value in metric_tracker.metrics.items()
        ),
    )

    return log_string, final_pred, final_prob, final_label


def run_epoch(epoch, model, training, data_loader, metric_tracker):
    iterations = epoch * (len(data_loader) )
    print(f"{len(data_loader) } batches.")

    final_pred = []
    final_label = []
    final_prob = []
    with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:
        # Clear this for every epoch. Only save the best epoch results.
        for x, y,seg,  ids in data_loader:
            if training:
                log_string = train_iter(
                    metric_tracker=metric_tracker,
                    model=model,
                    x=x,
                    y = y,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,

                )

            else:
                log_string, final_pred,final_prob,  final_label = eval_iter(
                    model=model,
                    x=x,
                    y=y,
                    ids = ids,
                    iteration=iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                    final_pred=final_pred,
                    final_prob = final_prob,
                    final_label=final_label
                )

            pbar.set_description(log_string)
            pbar.update(1)
            iterations += 1
    return final_pred, final_prob, final_label if not training else None


def save_model(saved_models_filepath, latest_epoch_filename, best_epoch_filename, is_best, model_name):
    if args.save:
        state = {
            "args": args,
            "epoch": epoch,
            "model": model.state_dict(),
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
        if is_best[model_name]:
            best_model_path = save_checkpoint(
                state=state,
                directory=saved_models_filepath,
                filename=best_epoch_filename,
                is_best=is_best[model_name],
            )
            epoch_pbar.set_description(
                "Saving best checkpoint at {}/{}".format(
                    saved_models_filepath, best_model_path
                )
            )
    return best_model_path


if __name__ == "__main__":
    loss_weight_dict = None

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

    # print(model.state_dict().keys())

    if args.resume:
        start_epoch = restore_model(restore_fields, path=saved_models_filepath, device=device)




    metric_tracker_train = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="training",
        task_dict=task_dict
    )
    metric_tracker_val = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="validation",
        task_dict=task_dict
    )

    metric_tracker_test = MetricTracker(
        metrics_to_track=metrics_to_track,
        load=True if start_epoch > 0 else False,
        path="",
        log_dict=log_dict,
        tracker_name="testing",
        task_dict=task_dict
    )

    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        best_acc = {}
        best_epoch = {}
        temp_best_epoch = {}
        current_val_acc = {}
        is_best = {}
        task_best_model_save_path = {}
        each_epoch_all_tasks_acc = []

        for task_name in task_dict.keys():
            best_acc[task_name] = 0.0
            best_epoch[task_name] = 0
            temp_best_epoch[task_name] = -1
            task_best_model_save_path[task_name] = ""
        best_acc["all_tasks"] = 0.0
        metric_tracker_train_log_dict_list = {}
        metric_tracker_val_log_dict_list = {}

        for epoch in range(start_epoch, args.max_epochs):
            print()
            print(f"Running epoch {epoch}")
            best_epoch_stats = open(logs_filepath + "/best_epoch_stats.txt", "w")
            run_epoch(
                epoch,
                data_loader=train_loader,
                model=model,
                training=True,
                metric_tracker=metric_tracker_train,
            )
            final_predictions,final_probabilities,  final_labels = run_epoch(
                epoch,
                data_loader=val_loader,
                model=model,
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

            current_val_acc = {}
            current_val_acc["all_tasks"] = 0.0
            is_best["all_tasks"] = False
            epoch_metric = metric_tracker_val.collect_per_epoch()
            for task_name in task_dict.keys():
                is_best[task_name] = False
                if not np.isnan(epoch_metric[f"{task_name}_accuracy_mean"][-1]):
                    current_val_acc[task_name] = epoch_metric[f"{task_name}_accuracy_mean"][-1]
                else:
                    current_val_acc[task_name] = 0.0
                latest_epoch_filename = f"{task_name}_latest_ckpt.pth.tar"

                best_epoch_filename = f"{task_name}_ckpt.pth.tar"
                current_val_acc["all_tasks"] += current_val_acc[task_name]
                if current_val_acc[task_name] >= best_acc[task_name]:
                    best_acc[task_name] = current_val_acc[task_name]
                    is_best[task_name] = True
                    best_epoch[task_name] = epoch
                    task_best_model_save_path[task_name] = save_model(saved_models_filepath=saved_models_filepath,
                                                                      latest_epoch_filename=latest_epoch_filename,
                                                                      best_epoch_filename=best_epoch_filename,
                                                                      is_best=is_best,
                                                                      model_name=task_name)
                else:
                    is_best[task_name] = False

            #         ############################################################################### Saving models
            #
            #     ##########################################################################################################
            final_predictions = np.array(final_predictions)
            final_probabilities = np.array(final_probabilities)
            final_labels = np.array(final_labels)
            temp = (final_predictions == final_labels)

            # print('AUC:', roc_auc_score(final_labels, final_probabilities))


            current_val_acc["all_tasks"] = float(sum(temp) / len(final_predictions))

            print(f"\ncurrent_val_acc['all_tasks']: {current_val_acc['all_tasks']:.4f}")
            latest_epoch_filename = "latest_ckpt.pth.tar"
            best_epoch_filename = "ckpt.pth.tar"
            if current_val_acc["all_tasks"] >= best_acc["all_tasks"]:
                best_acc["all_tasks"] = current_val_acc["all_tasks"]
                is_best["all_tasks"] = True
                best_epoch["all_tasks"] = epoch
                best_model_save_path = save_model(saved_models_filepath=saved_models_filepath,
                                                  latest_epoch_filename=latest_epoch_filename,
                                                  best_epoch_filename=best_epoch_filename,
                                                  is_best=is_best,
                                                  model_name="all_tasks"
                                                  )
                # eval_df.to_csv(eval_df_path,index=False)
            else:
                is_best["all_tasks"] = False
                save_model(saved_models_filepath=saved_models_filepath,
                           latest_epoch_filename=latest_epoch_filename,
                           best_epoch_filename=best_epoch_filename,
                           is_best=is_best,
                           model_name="all_tasks"
                           )
            print(
                "\nAll tasks till epoch: {} best_acc: {:.4f} best_epoch: {:}".format(
                    epoch, best_acc["all_tasks"], best_epoch["all_tasks"]
                )
            )
            epoch_pbar.set_description("")
            epoch_pbar.update(1)
            # Save important stats snapshot.
            # Track if best epoch changed, only compute best epoch stats if changed
            print()
            best_epoch_stats.write(f"Till epoch: {epoch}\n")
            for k in best_acc.keys():
                print(f"Task: {k} best_val_epoch: {best_epoch[k]}")
                print(f"Task: {k} best_val_acc: {best_acc[k]:.4f}")
                best_epoch_stats.write(f"Task: {k} best_val_epoch: {best_epoch[k]}\n")
                best_epoch_stats.write(f"Task: {k} best_val_acc: {best_acc[k]:.4f}\n")

            # plot each task at each epoch
            each_epoch_all_tasks_acc.append(current_val_acc["all_tasks"])
            metric_tracker_train.path = "{}/all_tasks_metrics_train.pt".format(logs_filepath.replace(os.sep, "/"))
            metric_tracker_train.save()
            metric_tracker_val.path = "{}/all_tasks_metrics_val.pt".format(logs_filepath.replace(os.sep, "/"))
            metric_tracker_val.save()
            # Save train log_dict
            metric_tracker_train_log_dict_list = metric_tracker_train.log_dict
            metric_tracker_train_log_dict_path = os.path.join(logs_filepath, "metric_tracker_train_log_dict.pt")
            save_metrics_dict_in_pt(path=metric_tracker_train_log_dict_path,
                                    metrics_dict=metric_tracker_train_log_dict_list,
                                    overwrite=True)
            metric_tracker_val_log_dict_list = metric_tracker_val.log_dict
            metric_tracker_val_log_dict_path = os.path.join(logs_filepath, "metric_tracker_val_log_dict.pt")
            save_metrics_dict_in_pt(path=metric_tracker_val_log_dict_path,
                                    metrics_dict=metric_tracker_val_log_dict_list,
                                    overwrite=True)

        task_best_model_save_path["lesion_side"], best_model_save_path, task_best_model_save_path["lesion_no_lesion"]
        best_model_save_path = f"{os.getcwd()}/models/pretrained_weights/best_fine_tune_MTL_ckpt.pth.tar"
        if len(task_dict) == 1 and list(task_dict.keys())[0] == "lesion_no_lesion":
            val_model_path = task_best_model_save_path["lesion_no_lesion"]
        elif len(task_dict) == 1 and list(task_dict.keys())[0] == "lesion_side":
            val_model_path = task_best_model_save_path["lesion_side"]
        elif len(task_dict) == 2:
            val_model_path = best_model_save_path

        best_epoch_stats.write(
            f"best_model path: {val_model_path}\n")
        restore_model_from_path(restore_fields, path=val_model_path, device=device)
        best_model.load_state_dict(restore_fields["model"].state_dict())


        resume_epoch = restore_model(
            restore_fields,
            path=saved_models_filepath,
            device=device,  # epoch=best_epoch_val_model_epoch (resume from latest checkpoint)
        )

    main_end = datetime.now()
    dt_string = main_end.strftime("%d/%m/%Y %H:%M:%S")
    print("End main() date and time =", dt_string)

    main_execute_time = main_end - main_start
    print("main() execute time: {}".format(main_execute_time))
    sys.exit(0)

