import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
      
class Dataset_from_folder(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, toGray = False):
            'Initialization'
            self.x = x
            self.transform = None if toGray == False else transforms.Compose([transforms.Grayscale(num_output_channels=1) ])
            self.convert_tensor = transforms.ToTensor()

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)

      def __getitem__(self, index):
            'Generates one sample of data'
            
            input = self.x[index]                 
            
            if self.x[index].shape[1] > self.x[index].shape[3]:
                  input = self.x[index].transpose( (0,3,1,2))

            input = torch.from_numpy(input)
            out = input         

            return input, out
            


def getLoader_folder(X, split = True, batch_size = 64):   

      if split:     
            #split dataset 80-20 for training and validation
            train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

            #create train and test dataloaders

            train_dataset = DataLoader( Dataset_from_folder(train_x), batch_size=batch_size, shuffle=True)
            val_dataset = DataLoader( Dataset_from_folder(val_x), batch_size=batch_size, shuffle=True)   
            
            #Comment previus two lines and uncomment next two lines 
            #if your dataset is small

            # train_dataset = DataLoader( Dataset_from_folder(X), batch_size=batch_size, shuffle=True)
            # val_dataset = DataLoader( Dataset_from_folder(X), batch_size=batch_size, shuffle=True)  

            return train_dataset, val_dataset, train_x, val_x 
      else :
            return DataLoader( Dataset_from_folder(X), batch_size=1, shuffle=False)

if __name__ == "__main__":
    pass
