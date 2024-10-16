from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import torch

class ImageCaptionDataset(Dataset):
    def __init__(self, data, resize):
        self.images = list(data.keys())
        self.captions = list(data.values())
        self.resize = resize
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 폴더가 데이터마다 다 달라서 구분하는 코드
        if self.images[idx].count('train2014') == 1:
            img_path = './images/train2014/' + self.images[idx]
        else:
            img_path = './images/val2014/' + self.images[idx]

        image = torch.tensor(np.array(cv2.resize(cv2.imread(img_path), self.resize)))
        caption = self.captions[idx]
        return image, caption

## 사용법 ##

# with open('./image_caption_dict.pkl', 'rb') as f:
#     train_data = pickle.load(f)

# dataset = ImageCaptionDataset(data=train_data, resize=RESIZE)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)