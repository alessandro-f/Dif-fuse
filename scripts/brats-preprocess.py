import os
import nibabel
import torch
import numpy as np
from torchvision import datasets, models, transforms
import imageio
import skimage

transform = transforms.Compose(
    [
        transforms.Pad(8)

    ]
)

directory = 'data/brats2021'


seqtypes = ['flair','t1', 't1ce', 't2', 'seg']
seqtypes_set = set(seqtypes)
database = []
for root, dirs, files in os.walk(directory):
    # if there are no subdirs, we have data
    if not dirs:
        files.sort()
        datapoint = dict()
        for f in files:
            f = f[:-7]
            seqtype = f.split('_')[2]
            datapoint[seqtype] = os.path.join(root, f+'.nii.gz')
        assert set(datapoint.keys()) == seqtypes_set, \
            f'datapoint {f} is incomplete, keys are {datapoint.keys()}'
        database.append(datapoint)

def load_all_levels(filedict):
    raw_image = []
    for level in  ['flair','t1','t2','t1ce']:
        nib_img = nibabel.load(filedict[level])
        niifti_data = nib_img.get_fdata()
        niifti_data = niifti_data.astype(np.float32)
        t = torch.tensor(niifti_data).permute([2, 0, 1])[25:-25,:,:]
        t = transform(t)

        raw_image.append(t/torch.max(t))
    nib_img = nibabel.load(filedict['seg'])
    niifti_data = nib_img.get_fdata()
    niifti_data = niifti_data.astype(np.float32)
    t = torch.tensor(niifti_data).permute([2, 0, 1])[25:-25, :, :]
    t = transform(t)
    raw_seg = t / torch.max(t)
    return raw_image, raw_seg


if not os.path.exists('data/brats2021_slices'):
    os.makedirs('data/brats2021_slices')
if not os.path.exists('data/brats2021_slices/images'):
    os.makedirs('data/brats2021_slices/images')
if not os.path.exists('data/brats2021_slices/segs'):
    os.makedirs('data/brats2021_slices/segs')

for i in range(len(database)):
    filedict = database[i]
    images, seg = load_all_levels(filedict)
    flair,t1, t2, t1ce = images
    number = os.sep.join(os.path.normpath(filedict['flair']).split(os.sep)[-2:])
    number = os.path.dirname(number)

    flair -= flair.min()
    flair /= flair.max()
    t1-= t1.min()
    t1 /= t1.max()
    t2 -= t2.min()
    t2 /= t2.max()
    t1ce -= t1ce.min()
    t1ce /= t1ce.max()
    seg -= seg.min()
    seg /= seg.max()


    for slice_index in range(flair.shape[0]):
        imageio.imwrite(f'data/brats2021_slices/images/{number}_{slice_index+25}_flair.png', skimage.img_as_ubyte(flair[slice_index, :, :]))
        imageio.imwrite(
            f'data/brats2021_slices/images/{number}_{slice_index+25}_t1.png',
            skimage.img_as_ubyte(t1[slice_index, :, :]))
        imageio.imwrite(
            f'data/brats2021_slices/images/{number}_{slice_index+25}_t2.png',
            skimage.img_as_ubyte(t2[slice_index, :, :]))
        imageio.imwrite(
            f'data/brats2021_slices/images/{number}_{slice_index+25}_t1ce.png',
            skimage.img_as_ubyte(t1ce[slice_index, :, :]))
        imageio.imwrite(f'data/brats2021_slices/segs/{number}_{slice_index+25}_seg.png', skimage.img_as_ubyte(seg[slice_index, :, :]))


