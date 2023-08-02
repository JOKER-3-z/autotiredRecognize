import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
from utils import AverageMeter, accuracy, accuracy_pskd
import numpy as np
import time
from loader import custom_datasets
from pskd_loss import Custom_CrossEntropy_PSKD
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

origin_pth = os.path.split(os.path.realpath(__file__))[0]
os.chdir(origin_pth)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="efficientnet-b7")
parser.add_argument("--pre_trained", type=bool, default=True)
parser.add_argument("--data_path", type=str, default="dataset")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--classes_num", type=int, default=2)
parser.add_argument("--dataset", type=str, default="sleepy", help="dataset")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--milestones", type=int, nargs="+", default=[10, 15])
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-6)
parser.add_argument("--mixup", type=bool, default=False)
parser.add_argument("--PSKD", type=bool, default=True)
parser.add_argument("--alpha", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=74)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=1)
parser.add_argument("--exp_postfix", type=str, default="efficientnet-b7_balanced_color_1e-5")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

exp_name = args.exp_postfix
exp_path = "./report/{}/{}".format(args.dataset, exp_name)
os.makedirs(exp_path, exist_ok=True)


#dataloader
if args.dataset == 'sleepy':
    transform_train = transforms.Compose([transforms.RandomRotation(90),
                                          transforms.Resize([256, 256]),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                 saturation=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), 
                                                               (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([transforms.RandomRotation(90),
                                         transforms.Resize([224, 224]),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomAffine(10),
                                         transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                 saturation=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), 
                                                              (0.229, 0.224, 0.225))
    ])

    trainset = custom_datasets.Custom_ImageFolder(root=os.path.join('../xfdata/sleepy', 'train'), 
                                                        transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join('../xfdata/sleepy', 'valid'),
                                                        transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                                   shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4,
                                                  shuffle=False, pin_memory=True)

else:
    raise NameError("The name of dataset isn't sleepy.")

#train

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
 
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
 
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
 
 
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, train_loader, epoch, scaler, PSKD, all_predictions,
                    mixup, criterion):
    model.train()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    if args.PSKD:
        alpha_t = args.alpha * ((epoch + 1) / args.epoch)
        alpha_t = max(0, alpha_t)

    for i, (inputs, targets, input_indices) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        
        if PSKD:
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes), dtype=torch.half) 
            targets_one_hot = identity_matrix[targets_numpy]
            
            if epoch == 0:
                all_predictions[input_indices] = targets_one_hot

            # create new soft-targets
            soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
            soft_targets = soft_targets.cuda()
                
            # compute output            
            with autocast():
                out = model(inputs)
                loss = criterion_CE_pskd(out, soft_targets)

        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            with autocast():
                out = model(inputs)
                loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)

        else:
            with autocast():
                out = model(inputs)
                loss = F.cross_entropy(out, targets)
        
        loss_recorder.update(loss.item(), n=inputs.size(0))
        acc = accuracy_pskd(out, targets)
        acc_recorder.update(acc, n=inputs.size(0))
        softmax_output = F.softmax(out, dim=1) 
        all_predictions[input_indices] = softmax_output.cpu().detach()
        
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)	
        scaler.update()	
        

    losses = loss_recorder.avg
    acces = acc_recorder.avg

    return losses, acces, all_predictions


def evaluation(model, test_loader):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in test_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            out = model(img)
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))
    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces


def train(model, optimizer, train_loader, test_loader, scheduler, PSKD):
    since = time.time()
    best_acc = -1
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    f = open(os.path.join(exp_path, "log_test.txt"), "w")
    all_predictions = torch.zeros(len(train_loader.dataset), 
                                  len(train_loader.dataset.classes), 
                                  dtype=torch.half)

    for epoch in range(args.epoch):
        train_losses, train_acces, all_predictions = train_one_epoch(
            model, optimizer, train_loader, epoch, scaler, PSKD, all_predictions,
            args.mixup, criterion
        )
        test_losses, test_acces = evaluation(model, test_loader)

        if test_acces > best_acc:
            best_acc = test_acces
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=test_acces)
            name = os.path.join(exp_path, args.model_names, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)

        scheduler.step()

        tags = ['train_losses',
                'train_acces',
                'test_losses',
                'test_acces']
        tb_writer.add_scalar(tags[0], train_losses, epoch+1)
        tb_writer.add_scalar(tags[1], train_acces, epoch+1)
        tb_writer.add_scalar(tags[2], test_losses, epoch+1)
        tb_writer.add_scalar(tags[3], test_acces, epoch+1)

        if (epoch + 1) % args.print_freq == 0:
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  test loss{:.2f} acc:{:.2f}\n".format(
                epoch+1,
                args.model_names,
                train_losses,
                train_acces,
                test_losses,
                test_acces,
            )
            print(msg)
            f.write(msg)
            f.flush()

    msg_best = "model:{} best acc:{:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "traninng time: {}".format(time.time() - since)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == "__main__":
    tb_path = "runs/{}/{}/{}".format(args.dataset, args.model_names,
                                            args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)
    lr = args.lr
    if 'efficient' in args.model_names:
        model = EfficientNet.from_pretrained(args.model_names)
        in_channel = model._fc.in_features
        model._fc = nn.Linear(in_channel, 2)
    else:
        raise NameError("The name of model should be efficientnet.")


    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda()
    scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    train(model, optimizer, train_loader, test_loader, scheduler, args.PSKD)