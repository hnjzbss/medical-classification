import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MIMICCXRDataset(Dataset):
    def __init__(self, dataframe, data_path=None, finding="any", transform=None):
        self.dataframe = dataframe
        self.data_path = data_path
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform

        if not finding == "any":
            if finding in self.dataframe.columns:
                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.PRED_LABEL = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                           "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
                           "Pneumonia", "Pneumothorax", "Support Devices"]

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        # 原始读图片
        """
        img = imread(os.path.join(self.dataset_path, str("{:06d}".format(item["image_new_id"])) + ".jpg"))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
        """
        if self.data_path is not None:
            path = self.data_path + item["path"].replace("MIMIC-CXR/files/", "")
        else:
            path = item["path"]
        try:
            img = Image.open(path)
        except:
            print(path)

        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        return img, label

    def __len__(self):
        return self.dataset_size
