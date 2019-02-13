import time
import numpy as np
from torchvision import transforms, datasets, models
import torch
from torch.utils.data import DataLoader
from torch import nn, optim


#data directories
rootdata = '../asr'
traindir = f'{rootdata}/train/'
testdir = f'{rootdata}/test/'
validdir = f'{rootdata}/valid/'

# non trainable params
batch_size = 4
n_epochs = 50
min_val_loss = np.Infinity
epochs_no_improve = 0
n_epochs_stop = 10

#other parameters
MODEL_PATH = 'checkpoints/best_model.mdl'

"""
aggregate the number of images using transformations
"""
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['valid'])
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

"""load resnet18 model pretrained"""
model = models.resnet18(pretrained=True)

#freeze parameters
for param in model.parameters():
	param.requires_grad = False

# 512 is coming from the preetrained model
# out_features are depending on our classes
model.fc = nn.Linear(in_features=512, out_features=2)

model = model.to('cuda')
model = nn.DataParallel(model)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(n_epochs):
	val_loss = 0
	start = time.time()

	#training
	for data, targets in dataloaders['train']:
		data = data.to('cuda')
		targets = targets.to('cuda')
		out = model(data)
		loss = criterion(out, targets)
		loss.backward()
		optimizer.step()
	
	#validating
	for data, targets in dataloaders['valid']:
		data = data.to('cuda')
		targets = targets.to('cuda')
		out = model(data)
		loss = criterion(out, targets)
		val_loss += loss

	#average validation loss
	val_loss = val_loss/len(dataloaders['train'])

	if val_loss < min_val_loss:
		torch.save(model, MODEL_PATH)
		epochs_no_improve = 0
		min_val_loss = val_loss
	else:
		epochs_no_improve += 1
		if epochs_no_improve == n_epochs_stop:
			print("early stopping")
			model = torch.load(MODEL_PATH)
			break
	
	end = time.time()
	print(f'epoch: {epoch} => loss: {val_loss}, time: %.2f'%(end-start))

val_accuracy = []
for data, targets in dataloaders['test']:
	data = data.to('cuda')
	log_ps = model(data)
	targets = targets.to('cuda')
	#convert predictions to probabilities
	ps = torch.exp(log_ps)
	pred = torch.argmax(ps, dim=1)
	equals = pred == targets
	equals = torch.tensor(equals, dtype=torch.float32)
	accuracy = torch.mean(equals)
	val_accuracy.append(accuracy)

