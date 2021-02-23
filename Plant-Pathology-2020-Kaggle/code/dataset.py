from torch.utils.data import Dataset
from PIL import Image
import torch


class ppDataset(Dataset):
    def __init__(self, df, image_dir, return_labels=False, transforms=None):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.return_labels = return_labels
        self.label_map = {'healthy':0, 'multiple_diseases':1, 'rust':2, 'scab':3}
        self.label_map_reverse = {v:k for k,v in self.label_map.items()}
        
    def __len__(self):
        return self.df.__len__()
    
    def __getitem__(self, idx):
        image_path = self.image_dir + self.df.loc[idx, 'image_id'] + '.jpg'
        image = Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)

        if self.return_labels:
            # label = torch.tensor(self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']]).unsqueeze(1)
            label = torch.tensor(self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']]).unsqueeze(-1)
            label_flatten = label.argmax()
            return image, label_flatten, self.label_map_reverse[label.squeeze(1).numpy().argmax()]
        else:
            return image
