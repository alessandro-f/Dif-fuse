import torch
import math
import sys
import torchvision




class SliceSamplingUniform(object):

    def __init__(self, total_slices):
        self.total_slices = total_slices

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (S, C, H, W)

        Returns:
            Tensor: Tensor image with self.total_slices slices
        """

        assert torch.is_tensor(tensor), (
            "Input object is not a tensor, please convert to tensor before calling on this "
            "function"
        )

        slice_number = tensor.shape[0]
        if slice_number == self.total_slices:
            slice_index = list(range(self.total_slices))
        elif slice_number > self.total_slices:
            middle_slice = math.floor(slice_number / 2.0)
            intervals = math.floor(self.total_slices / 2.0 + 1)
            increment = math.floor(slice_number / 2.0 / intervals)
            start = middle_slice - increment * (intervals - 1)
            end = middle_slice + increment * (intervals - 1)
            slice_index = list(range(start - 1, end, increment))
        else:
            print(
                "slice number {} less than target slices {}".format(
                    slice_number, self.total_slices
                )
            )
            sys.exit(1)

        sampled_slice_tensor = tensor[slice_index, :, :, :]

        return sampled_slice_tensor






