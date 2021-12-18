import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import os

classes = os.listdir('./training_dataset')

model = torch.load('best_model.pth')
mean = [0.5081, 0.5185, 0.4952]
std = [0.2001, 0.2005, 0.2181]

image_transforms = transforms.Compose([
    transforms.resize((512,910)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).cuda()
    image = image.unsqueeze(0)

    output = model(image)
    _, prediction = torch.max(output.data,1)

    print(classes[prediction.item()])

classify(model, image_transforms, 'coca_cola.jpg', classes)
