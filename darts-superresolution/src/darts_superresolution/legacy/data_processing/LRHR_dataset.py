import random
from io import BytesIO

import lmdb
import tifffile
from PIL import Image
from torch.utils.data import Dataset

import darts_superresolution.data_processing.util as Util


class LRHRDataset(Dataset):
    def __init__(
        self, dataroot, datatype, l_resolution=16, r_resolution=128, split="train", data_len=-1, need_LR=False
    ):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        # print("Length", data_len)
        if datatype == "lmdb":
            self.env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get(b"length"))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == "img":
            self.sr_path = Util.get_paths_from_images(f"{dataroot}/sr_{l_resolution}_{r_resolution}")
            self.hr_path = Util.get_paths_from_images(f"{dataroot}/hr_{r_resolution}")
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(f"{dataroot}/lr_{l_resolution}")
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(f"data_type [{datatype:s}] is not recognized.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == "lmdb":
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(f"hr_{self.r_res}_{str(index).zfill(5)}".encode())
                sr_img_bytes = txn.get(f"sr_{self.l_res}_{self.r_res}_{str(index).zfill(5)}".encode())
                if self.need_LR:
                    lr_img_bytes = txn.get(f"lr_{self.l_res}_{str(index).zfill(5)}".encode())
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len - 1)
                    hr_img_bytes = txn.get(f"hr_{self.r_res}_{str(new_index).zfill(5)}".encode())
                    sr_img_bytes = txn.get(f"sr_{self.l_res}_{self.r_res}_{str(new_index).zfill(5)}".encode())
                    if self.need_LR:
                        lr_img_bytes = txn.get(f"lr_{self.l_res}_{str(new_index).zfill(5)}".encode())
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # print("HR Path: ", self.hr_path[index], "SR PAth: ", self.sr_path[index])
            img_HR = tifffile.imread(self.hr_path[index])  # Image.open(self.hr_path[index])#.convert("RGB")
            # img_HR = img_HR.transpose(2, 0, 1)

            img_SR = tifffile.imread(self.sr_path[index])  # Image.open(self.sr_path[index])#.convert("RGB")
            img_SR = img_SR.transpose(1, 2, 0)

            # just 4th channel

            # img_HR = np.expand_dims(img_HR[:, :, 3], axis=2)
            # img_SR = np.expand_dims(img_SR[:, :, 3], axis=2)

            # print("Only 4th channel: ", img_HR.shape, img_SR.shape)

            # img_HR = img_HR[:,:,(0,0,0)]
            # img_SR = img_SR[:,:,(0,0,0)]

            # print("As 3 channels: ", img_HR.shape, img_SR.shape)

            # print("SR: ", img_SR.dtype)
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index])  # .convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1)
            )
            return {"LR": img_LR, "HR": img_HR, "SR": img_SR, "Index": index}
        else:
            # Image to tensor:
            # print("Before tensor: ", img_SR.min(), img_SR.max(), img_HR.min(), img_HR.max())
            [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
            # print("After tensor: ", img_SR.min(), img_SR.max(), img_HR.min(), img_HR.max())

            # These images look fine
            # if self.split=="val":
            # print("Val_path: ", self.sr_path[index])
            # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_sr_2_"+os.path.basename(self.hr_path[index]), np.asarray(img_SR))
            # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_hr_2_"+os.path.basename(self.sr_path[index]), np.asarray(img_HR))
            return {"HR": img_HR, "SR": img_SR, "Index": index}
