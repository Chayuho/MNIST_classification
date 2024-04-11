
# import some packages you need here
from torch.utils.data import Dataset, DataLoader
import glob as glob
from PIL import Image
import numpy as np
from torchvision import transforms

torchvision_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class MNIST(Dataset):


    def __init__(self, data_dir, transform = None, model = None):
        
        # write your codes here
        self.data_path = data_dir # path your dir
        self.img_name = glob.glob(self.data_path+"/*.png")
        self.label = [int(f_name.split("_")[-1][0]) for f_name in self.img_name]
        self.transform = transform
        self.model_name = model
    def __len__(self):

        # write your codes here
        return len(self.img_name)

    def __getitem__(self, idx):
    
        # write your codes here
        img = Image.open(self.img_name[idx])
        img = self.transform(img)
        if self.model_name == "LeNet5":
            img = np.array(img)
        elif self.model_name == "CustomMLP":
            img = np.array(img)
            img = img.reshape(1, -1)
            
            
        label = self.label[idx]

        # return torch.tensor(img), torch.LongTensor(int(label))
        return img, label

def get_loader(dir_path, batch, train_or_test = None, model = None):
    dataset = MNIST(dir_path, transform = torchvision_transform, model = model)
    if train_or_test == "train":
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers = 16)
    if train_or_test == "test":
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers = 16)
    return loader

if __name__ == '__main__':

    # write test codes to verify your implementations
    print("load image ...")

