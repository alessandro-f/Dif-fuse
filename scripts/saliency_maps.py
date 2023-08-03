import sys
# put your path here
sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
import numpy as np
from utils.arg_parsing import parse_args
from datetime import datetime
from torch.autograd import grad
import random
from autoencoder_architectures import *
import torchvision
from guided_diffusion.brain_datasets import *
from torch.utils.data import DataLoader
import imageio
import skimage
from utils.storage import (
    build_experiment_folder,
    restore_model,
)

################################################################################## Data


args = parse_args()

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

################################################################################## Model

model = torchvision.models.resnet50(progress=False)
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
model.fc = nn.Linear(2048, 2)

model = model.to(device)
if args.num_gpus_to_use > 1:
    model = nn.DataParallel(model)

baseline_filepath, _, _ = build_experiment_folder(
    experiment_name='baseline_classifier',
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),

)
_ = restore_model(restore_fields={"model": model}, path=baseline_filepath, device=device, best=True)

model.eval()

enc_out = 512
ae = AE_no_bottleneck_6_12_16(batch_size= args.batch_size, input_height = 256, enc_type='my_resnet18', first_conv=False, maxpool1=False, enc_out_dim=enc_out, latent_dim=int(enc_out/2), lr=0.0001)

ae = ae.to(device)
if args.num_gpus_to_use > 1:
    ae = nn.DataParallel(ae)
    ae.module.encoder = nn.DataParallel(ae.module.encoder)
    ae.module.decoder = nn.DataParallel(ae.module.decoder)

autoencoder_filepath, _, _ = build_experiment_folder(
    experiment_name='autoencoder',
    log_path=args.logs_path + "/" + "epochs_" + str(args.max_epochs),

)
_ = restore_model(restore_fields={"model": ae}, path=autoencoder_filepath, device=device, best=True)

ae.eval()

########################################################################## Optimisation


def to_tensor_grad(z, requires_grad=False):
    z = torch.Tensor(z).to(device)
    z.requires_grad=requires_grad
    return z

def to_numpy(z):
    return z.data.cpu().numpy()

alpha = 100
beta = 0.001
m = nn.Softmax(dim=-1)
def compute_counterfactual(z, z0, targets, criterion_class = nn.CrossEntropyLoss(),  criterion_norm = nn.L1Loss()):
    for i in range(20):
        # print(i)
        z = to_tensor_grad(z, requires_grad=True)
        z_0 = to_tensor_grad(z0)
        logits = model(ae.module.decoder(z))
        saliency_loss = criterion_class(input=logits, target=targets)
        distance = criterion_norm(z, z_0)
        # print('loss', saliency_loss, 'distance',distance, 'prob', m(logits)[:, 1])

        loss = saliency_loss + alpha * distance

        dl_dz = grad(loss, z)[0]
        z = z - beta * dl_dz
        z = to_numpy(z)
        # print(z)
    return to_tensor_grad(z)

def compute_saliency(z, im2):
    shifted_image_1 = ae.module.decoder(z)
    shifted_image_1 = shifted_image_1.detach().cpu()
    dimage = torch.abs(im2.cpu()- shifted_image_1)

    return dimage



saliency_root = 'saliency_maps'

if not os.path.exists(saliency_root):
    os.makedirs(saliency_root)


for loader in [train_loader, val_loader, test_loader]:
    for i, (inputs, _,_, ids) in enumerate(loader):
            name = ids
            print(name)
            inputs = inputs.to(device)

            im1_enc = inputs
            im2 = ae.module.decoder(ae.module.encoder(im1_enc)).detach().to('cpu')

            z = to_numpy(ae.module.encoder(im1_enc))
            z0 = z

            # positive counterfactual
            targets = (torch.ones([inputs.shape[0]], dtype=torch.long)).to(device)
            z_out = compute_counterfactual(z.copy(), z0.copy(), targets)
            dimage1 = compute_saliency(z_out, im2.clone())

            # negative counterfactual
            targets = (torch.zeros([inputs.shape[0]], dtype=torch.long)).to(device)
            z_out = compute_counterfactual(z.copy(), z0.copy(), targets)
            dimage2 = compute_saliency(z_out, im2.clone())


            dimage1 = dimage1*(1.0 / torch.amax(dimage1, dim=(-3, -2, -1), keepdim=True))
            dimage2 = dimage2*(1.0 / torch.amax(dimage2, dim=(-3, -2, -1), keepdim=True))

            dimage = (dimage1+dimage2)/2
            dimage = dimage*(1.0 / torch.amax(dimage, dim=(-3, -2, -1), keepdim=True))

            for j in range(inputs.shape[0]):
                for i, level in enumerate(['flair', 't1', 't2', 't1ce']):
                    path = os.path.join(saliency_root, name[j][:-9] + level + '.png')
                    imageio.imwrite(path, skimage.img_as_ubyte(dimage[j,i, :, :]))



