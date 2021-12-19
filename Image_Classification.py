import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import sys
import os
import json

classes = ['7up', 'AMP_365_Energy', 'AMP_365_Forte', 'AMP_365_Recargado', 'California_Kiwi_Fresa',
           'California_Mora_Uva', 'California_Pina_Coco',
           'Canada_Dry_Ginger_Ale', 'Coca_Cola', 'Coca_Cola_Sabor_Ligero', 'Dasani', 'Enjoy_Guava_Pina',
           'Enjoy_Naranja', 'Enjoy_Pera', 'Fresca',
           'Fury_Energy_Gold_Strike', 'Fury_Energy_Mean_Green', 'Gatorade_Berry_Blue', 'Gatorade_Fruit_Punch',
           'Gatorade_Lemon_Lime', 'Gatorade_Mango_Verde',
           'Gatorade_Melon', 'Gatorade_Naranja', 'Gatorade_Uva', 'H2_OH_Limoneto', 'Link_Banana', 'Link_Coco',
           'Link_Mandarina', 'Lipton_Durazno', 'Lipton_Frambuesa',
           'Lipton_Limon', 'Mirinda_Banana', 'Mirinda_Naranja', 'Mirinda_Uva', 'Mountain_Dew', 'Pepsi', 'Pepsi_Blue',
           'Pepsi_Light', 'Raptor', 'Salutaris', 'Sprite',
           'Tropical_Banana', 'Tropical_Naranja', 'Tropical_Uva', 'Zen']

mean = [0.5081, 0.5185, 0.4952]
std = [0.2001, 0.2005, 0.2181]

image_transforms = transforms.Compose([
    transforms.Resize((512, 910)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).cuda()
    image = image.unsqueeze(0)

    output = model(image)
    _, prediction = torch.max(output.data, 1)
    return classes[prediction.item()]

def updateDictionary(i, marca):
    dictionary.update({i: {
        "marca": marca
    }})

def createJson():
    json_obj = json.dumps(dictionary, indent=4)
    with open("predictions.json", "w") as outfile:
        outfile.write(json_obj)


def main():
    for i in os.listdir(directory):
        marca = classify(model, image_transforms, directory + i, classes)
        updateDictionary(i, marca)
    createJson()


if __name__ == '__main__':
    directory = sys.argv[1]
    model_path = sys.argv[2]
    model = torch.load(model_path)
    dictionary = {}
    main()
