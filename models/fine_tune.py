import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import h5py, json

class VGImageDataset(Dataset):

    def __init__(self, imdb_h5, sgg_h5, sgg_dict, transform=None, target_transform=None):
        self.imdb = h5py.File(imdb_h5)
        self.sgg = h5py.File(sgg_h5)
        with open(sgg_dict) as f:
            self.dicts = json.load(f)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.imdb['images'])
    
    def __getitem__(self, idx):
        image = self.imdb['images'][idx]
        if self.transform:
            image = self.transform(torch.tensor(image))
        image.to(torch.device('cuda'))

        start = self.sgg['img_to_first_box'][idx]
        end = self.sgg['img_to_last_box'][idx]
        if start < 0:
            objects = [] # Some entries have a large negative value for both start & end
        else:
            objects = [i for i in range(start, end+1)]

        # Max objects is 40
        #labels = torch.full((1, 49), -1, dtype=float) # Use -1 for missing label
        labels = torch.tensor([-1 for _ in range(40)], dtype=float)
        for i in range(len(objects)):
            label = self.sgg['labels'][objects[i]][0]/100 # Make values smaller to expedite learning
            labels[i] = label
        
        if self.target_transform:
            labels = self.target_transform(labels)
        labels.to(torch.device('cuda'))

        return image, labels
    
    def colate_fn(self, data):
        batch = [d for d in data if d != None]
        return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 40)
    resnet.to(torch.device('cuda'))
    #print(net)

    params = [param for param in resnet.parameters()]
    for param in params:
        param.requires_grad = False
    params[-1].requires_grad = True
    
    imdb_path = '/home/grav/PRProject/mini-vg/mini_imdb_1024.h5'
    sgg_path = '/home/grav/PRProject/mini-vg/mini_VG-SGG.h5'
    sgg_dict_path = '/home/grav/PRProject/mini-vg/mini_VG-SGG-dicts.json'
    train_data = VGImageDataset(imdb_path, sgg_path, sgg_dict_path, transform=weights.transforms())

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 32

    train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=False, collate_fn=train_data.colate_fn)
    optimizer = torch.optim.SGD([params[-1]], lr=LEARNING_RATE, momentum=0.9)
    critereon = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_dataloader, 0):
            train_images, train_labels = data
            train_images = train_images.to(torch.device('cuda'))
            train_labels = train_labels.to(torch.device('cuda'))

            predictions = resnet(train_images)

            loss = critereon(predictions, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'[{epoch+1}] {i+1} loss: {loss.item()}')