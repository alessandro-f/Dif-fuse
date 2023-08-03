import pandas as pd
import os
import torch
import torch.utils.data as data
import imageio



class BRATSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root_folder_filepath,
            df_path,
            transform = None,
            only_positive = False,
            only_negative = False,
            only_flair = False

    ):
        super(BRATSDataset, self).__init__()
        self.dataset_root_folder_filepath = dataset_root_folder_filepath
        self.df_path = df_path
        self.only_positive = only_positive
        self.only_negative = only_negative
        self.transform = transform
        self.only_flair = only_flair

        self.meta_data_data_frame = pd.read_csv(
            self.df_path, encoding="ISO-8859-1"
        )
        if self.only_positive:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==1]
        if self.only_negative:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==0]

        self.sample_idx_to_scan_path_and_label = []

        self.sample_idx_to_scan_path_and_label = [
            (row["flair"], row["label"])  # Note we are using AIS lesion label here.
            for idx, row in self.meta_data_data_frame.iterrows()
        ]


    def __len__(self):

        return len(self.sample_idx_to_scan_path_and_label)

    def __getitem__(self, item):
        x_path, y_sample = self.sample_idx_to_scan_path_and_label[item]  # example: 1085

        raw_image = []
        for level in ['flair', 't1', 't2', 't1ce']:
            im = os.path.join(self.dataset_root_folder_filepath, x_path[:-9] + level + '.png')
            image = imageio.imread(im)
            image = torch.tensor(image, dtype=torch.float32)
            if self.transform is not None:
                image = self.transform(image)
            image = (image / torch.max(image))
            raw_image.append(image)
        im = torch.stack(raw_image)
        seg = os.path.join(self.dataset_root_folder_filepath[:-6] + 'segs', x_path[:-9] + 'seg' + '.png')
        seg = imageio.imread(seg)
        seg = torch.tensor(seg).unsqueeze(0)
        seg = (seg / torch.max(seg))
        if torch.max(seg) > 0:
            weak_label = 1
        else:
            weak_label = 0

        out_dict = {}
        out_dict["y"] = weak_label
        if self.only_flair:
            im = ((im[0, :, :]).unsqueeze(0))

        # return im, out_dict, seg, x_path
        return im, weak_label, seg, x_path

class BRATSDatasetSaliency(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root_folder_filepath,
            saliency_root_folder_filepath,
            df_path,
            transform = None,
            only_positive = False,
            only_negative = False,
            only_flair = False

    ):
        super(BRATSDatasetSaliency, self).__init__()
        self.dataset_root_folder_filepath = dataset_root_folder_filepath
        self.saliency_root_folder_filepath = saliency_root_folder_filepath
        self.df_path = df_path
        self.only_positive = only_positive
        self.only_negative = only_negative
        self.transform = transform
        self.only_flair = only_flair

        self.meta_data_data_frame = pd.read_csv(
            self.df_path, encoding="ISO-8859-1"
        )
        if self.only_positive:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==1]
        if self.only_negative:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==0]

        self.sample_idx_to_scan_path_and_label = []

        self.sample_idx_to_scan_path_and_label = [
            (row["flair"], row["label"])  # Note we are using AIS lesion label here.
            for idx, row in self.meta_data_data_frame.iterrows()
        ]


    def __len__(self):

        return len(self.sample_idx_to_scan_path_and_label)

    def __getitem__(self, item):
        x_path, y_sample = self.sample_idx_to_scan_path_and_label[item]  # example: 1085

        raw_image = []
        raw_sal = []
        for level in ['flair', 't1', 't2', 't1ce']:
            im = os.path.join(self.dataset_root_folder_filepath, x_path[:-9] + level + '.png')
            sal = os.path.join(self.saliency_root_folder_filepath, x_path[:-9] + level + '.png')
            image = imageio.imread(im)
            image = torch.tensor(image, dtype=torch.float32)
            sal = imageio.imread(sal)
            sal = torch.tensor(sal, dtype=torch.float32)
            if self.transform is not None:
                image = self.transform(image)
            image = (image / torch.max(image))
            sal = (sal/ torch.max(sal))
            raw_image.append(image)
            raw_sal.append(sal)
        im = torch.stack(raw_image)
        sal = torch.stack(raw_sal)

        seg = os.path.join(self.dataset_root_folder_filepath[:-6] + 'segs', x_path[:-9] + 'seg' + '.png')
        seg = imageio.imread(seg)
        seg = torch.tensor(seg).unsqueeze(0)
        seg = (seg / torch.max(seg))
        if torch.max(seg) > 0:
            weak_label = 1
        else:
            weak_label = 0

        out_dict = {}
        out_dict["y"] = weak_label
        if self.only_flair:
            im = ((im[0, :, :]).unsqueeze(0))

        # return im, out_dict, seg, x_path
        return im, weak_label, seg,sal, x_path





