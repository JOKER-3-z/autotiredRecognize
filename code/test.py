import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np
import pandas as pd
import shutil
import glob
import os
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

origin_pth = os.path.split(os.path.realpath(__file__))[0]
os.chdir(origin_pth)
shutil.move('../xfdata/sleepy/test', '../xfdata/sleepy/pred')

def predict(test_loader, model):
    model.eval()

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            test_pred.append(output.data.cpu().numpy())
            
    return np.vstack(test_pred)

#--------------
# load model
#--------------

from efficientnet_pytorch import EfficientNet
import torch.nn as nn

model = EfficientNet.from_name('efficientnet-b7')
in_channel = model._fc.in_features
model._fc = nn.Linear(in_channel, 2)
mpath = '../user_data/model_data/efficientnet-b7_balanced_color_1e-5/efficientnet-b7/ckpt/best.pth'
checkpoint = torch.load(mpath)
model.load_state_dict(checkpoint['model'])
model = model.cuda()

#-------------------
# TTA10
#-------------------

for i in range(10):
    transform_test = transforms.Compose([transforms.RandomRotation(90),
                                    transforms.Resize([224, 224]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                            saturation=0.5), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), 
                                                         (0.229, 0.224, 0.225))
    ])
    testset = datasets.ImageFolder(root='../xfdata/sleepy/pred',
                                transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=4
                                          )
    if i == 0:
        pred = predict(test_loader, model)
        pred = np.expand_dims(pred, axis=2)
    else:
        pred1 = predict(test_loader, model)
        pred1 = np.expand_dims(pred1, axis=2)
        pred = np.concatenate([pred, pred1], axis=2)
final_pred = np.sum(pred, axis=2)

#---------------
# result
#---------------

shutil.move('../xfdata/sleepy/pred/test', '../xfdata/sleepy')
test_path = glob.glob('../xfdata/sleepy/test/*')
submit = pd.DataFrame(
    {
        'id': [x.split('/')[-1].split('.')[0] for x in test_path],   
        'label': [['non-sleepy', 'sleepy'][x] for x in final_pred.argmax(1)]
})
#submit.to_csv('./res.csv')
submit['id'] = submit['id'].astype('int')
submit = submit.sort_values(by='id')
submit['label'].to_csv('../prediction_result/result.csv', 
                       index=None, header=None)