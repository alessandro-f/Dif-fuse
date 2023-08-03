import time
import os
import GPUtil


def set_gpu_visible_devices(num_gpus_to_use):

    if num_gpus_to_use == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpu_to_use = GPUtil.getAvailable(
            order="first",
            limit=num_gpus_to_use,
            maxLoad=0.1,
            maxMemory=0.1,
            includeNan=False,
            excludeID=[],
            excludeUUID=[],
        )

        if len(gpu_to_use) < num_gpus_to_use:
            print("Could not find enough GPU(s), waiting and retrying")
            time.sleep(1)
            return set_gpu_visible_devices(num_gpus_to_use=num_gpus_to_use)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu_idx) for gpu_idx in gpu_to_use]
        )

        print("GPUs selected have IDs {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
