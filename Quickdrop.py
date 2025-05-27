import argparse
import copy
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage.interpolation import rotate as scipyrotate
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)
    for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}



def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x

def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def get_info(dataset):
    if dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100

    elif dataset == 'EUROSAT':
        channel = 3
        im_size = (32, 32)  # resize (64, 64)
        num_classes = 10

    else:
        exit('unknown dataset: %s'%dataset)
    return channel, im_size, num_classes


class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def init_model(model, channel, num_classes, im_size=(32, 32)):
    # torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
    else:
        device = 'cpu'
    net = net.to(device)
    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def init_env(strategy, dataset, env, root_path):
    return os.path.join(root_path, strategy, dataset, env)


def train_loop(device: str, dataloader, model, loss_fc, optimizer):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    num_training_samples = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        num_training_samples += y.shape[0]
        pred = model(X).to(device)
        loss = loss_fc(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"\t{get_time()} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return num_training_samples


def test_loop(device: str, dataloader, model, loss_fc, num_classes):
    model.eval()
    test_loss, correct = 0.0, 0.0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fc(pred, y).item()

            predicted = pred.argmax(1)
            correct += (predicted == y).sum().item()

            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    test_loss /= num_batches
    overall_accuracy = correct / size

    per_class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
        else:
            acc = 0.0
        per_class_accuracy[i] = acc

    return overall_accuracy, test_loss, per_class_accuracy


class FedAvgServer:
    def __init__(self, args):
        self.args = args
        # load scenario
        self.device = args.device
        self.args.device = self.device
        self.dataset = args.dataset
        self.save_path = args.save_path

        # load properties of original dataset
        channel, im_size, num_classes = get_info(args.dataset)
        self.channel = channel
        self.im_size = im_size
        self.num_classes = num_classes

        # write statistics back
        args.channel = channel
        args.im_size = im_size
        args.num_classes = num_classes

        # initialize the global model
        self.model = args.model
        self.global_model = init_model(args.model, channel, num_classes, im_size).to(self.device)  # get a random model
        self.global_model.to(self.device)

        # initialize the fed env
        self.strategy = args.strategy
        self.env = args.env
        self.fed_env = init_env(args.strategy, args.dataset, args.env, args.env_path)

        if os.path.exists(self.fed_env):
            print('{} {}'.format(get_time(), 'Find fl env: {}'.format(self.fed_env)))
        else:
            raise Exception('{} No such fl env at: {}'.format(get_time(), self.fed_env))

        # initialize the clients
        self.users = []
        self.bz = args.batch_size
        self.user_trainsets = {}
        self.clean_test = None
        self.poisoned_test = None
        self.load_dataset()
        self.communication_round = args.communication_round
        self.clean_test_loader = DataLoader(self.clean_test,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=self.args.num_workers,
                                      pin_memory=self.args.pin_memory,
                                      persistent_workers=self.args.persistent_workers)
        
        self.poisoned_test_loader = DataLoader(self.poisoned_test,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=self.args.num_workers,
                                pin_memory=self.args.pin_memory,
                                persistent_workers=self.args.persistent_workers)
        
        self.client_instances = {uid: FedAvgClient(self.args, uid, self.device, self.user_trainsets[uid]) for uid in self.users}
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.local_epoch = args.local_epoch

        # records
        self._loss_fc = nn.CrossEntropyLoss()   # for testing
        self.global_model_records  = {}

    # Global update
    def weighted_global_update(self, n=-1):

        time_cnt = 0
        for r in range(self.communication_round):
            # print info
            print(f'{get_time()} Global Round {r}')
            # select participants:
            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users
            time_start = time.time()
            self.execute_update(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)
            avg_state_dict, avg_loss = self.weighted_aggregate(participants)
            print(f'Averaged train loss:{avg_loss:.4f}')
            self.update_global_model(avg_state_dict)
            overall_acc_clean, loss_clean, per_class_acc_clean, overall_acc_poison, loss_poison, per_class_acc_poison = self.test_global_model()
            self.global_model_records[len(self.global_model_records)] = {
                'train_loss': avg_loss,
                'overall_acc_clean': overall_acc_clean,
                'loss_clean': loss_clean,
                'per_class_acc_clean': per_class_acc_clean,
                'overall_acc_poison': overall_acc_poison,
                'loss_poison': loss_poison,
                'per_class_acc_poison': per_class_acc_poison
            }

        self.save()
        root = os.path.join(self.save_path, f'{self.args.dataset}-{self.args.env}-r{self.args.communication_round}')
        os.makedirs(root, exist_ok=True)
        accuracy_filename = f'results-train.pkl'
        accuracy_filepath = os.path.join(root, accuracy_filename)
        with open(accuracy_filepath, 'wb') as f:
            pickle.dump(self.global_model_records, f)
        print(f'{get_time()} Global Update Finished Using {time_cnt}')


    def load_check_point(self, ckpt: str):
        self.global_model.load_state_dict(torch.load(ckpt, weights_only=False).state_dict())

    def test_global_model(self):
        overall_acc_clean, loss_clean, per_class_acc_clean = test_loop(self.device, self.clean_test_loader, self.global_model, self._loss_fc, self.num_classes)
        overall_acc_poison, loss_poison, per_class_acc_poison = test_loop(self.device, self.poisoned_test_loader, self.global_model, self._loss_fc, self.num_classes)
        print(f"Clean Accuracy for global model: {overall_acc_clean:.4f}, loss: {loss_clean:.4f}")
        print(f"Backdoor Accuracy for global model: {overall_acc_poison:.4f}, loss: {loss_poison:.4f}")
        return overall_acc_clean, loss_clean, per_class_acc_clean, overall_acc_poison, loss_poison, per_class_acc_poison

    def update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def execute_update(self, participants: list):
        for client in participants:
            self.client_instances[client].local_update(self.local_epoch, self.global_model, self.lr, self.weight_decay, self.momentum, use_affine=False)

    def execute_recover_update(self, recover_epoch, participants: list):
        for client in participants:
            self.client_instances[client].local_update(recover_epoch, self.global_model, self.lr, self.weight_decay, self.momentum, use_affine=True)

    def weighted_aggregate(self, participants: list):
        usr_state_dicts = [self.client_instances[client].local_model.state_dict() for client in participants]
        avg_loss = np.mean([self.client_instances[client].loss_records[-1] if self.client_instances[client].loss_records else 0 
                    for client in participants])
        
        # calculating weights
        sample_nums = torch.tensor([
            len(self.client_instances[client].trainset) for client in participants
        ], dtype=torch.float).to(self.device)
        weights = sample_nums / sample_nums.sum()
        print(f'{get_time()} Weighted Aggregate for {len(participants)} participants, weights: {weights}')

        assert len(weights) == len(usr_state_dicts)
        avg_state_dict = copy.deepcopy(usr_state_dicts[0])
        for key in avg_state_dict.keys():
            avg_state_dict[key] = 0
            for i in range(0, len(participants)):
                avg_state_dict[key] += usr_state_dicts[i][key] * weights[i]
        return avg_state_dict, avg_loss

    # load scenario
    def load_dataset(self):
        self.load_train_dataset()
        self.load_test_dataset()

    def load_train_dataset(self):
        train_cluster = torch.load('{}/{}/{}'.format(self.fed_env, 'train', 'train.pt'))
        self.users = train_cluster['users']
        for usr in self.users:
            tmp = train_cluster['user_data'][usr]
            self.user_trainsets[usr] = TensorDataset(tmp['x'], tmp['y'])

    def load_test_dataset(self):
        test_cluster = torch.load('{}/{}/{}'.format(self.fed_env, 'test', 'test.pt'))
        self.clean_test = test_cluster['clean_test']
        self.poisoned_test = test_cluster['poisoned_test']

    def save(self):
        print("Saving model and results...")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

class FedAvgClient:
    def __init__(self,
                 args,
                 id,
                 device,
                 trainset,
                 ):
        self.id = id
        self.args = args
        self.trainset = trainset
        self.train_dataloader = DataLoader(trainset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=self.args.num_workers,
                                           pin_memory=self.args.pin_memory,
                                           persistent_workers=self.args.persistent_workers)

        self.device = device
        self.local_model = None
        self.epoch = None
        self.loss_fc = nn.CrossEntropyLoss()
        self.loss_records = []
        
    # local update
    def local_update(self, epoch, model, lr, weight_decay, momentum, use_affine=False):
        # print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        self.local_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        if use_affine:
            train_dataset = TensorDataset(self.image_syn_train, self.label_syn_train)
        else:
            train_dataset = self.train_dataloader.dataset

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)

        epoch_loss = []
        for i in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.local_model.zero_grad()
                log_probs = self.local_model(images)
                loss = self.loss_fc(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.loss_records.append(sum(epoch_loss) / len(epoch_loss))
        # print('{} {}'.format(get_time(), 'Client {} completes update'.format(self.id)))



class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


class FedQuickDropServer(FedAvgServer):
    def __init__(self, args):
        super().__init__(args)
        self.args.dsa_param = ParamDiffAug()
        self.client_instances = {uid: FedQuickDropClient(self.args, uid, self.device, self.user_trainsets[uid]) for uid in self.users}

    def weighted_global_update(self, n=-1):
        time_cnt = 0

        for r in range(self.communication_round):
            print(f'{get_time()} Global Round {r}')

            if n != -1:
                participants = random.sample(self.users, n)
                print(f'{get_time()} Select {participants} execute local updates.')
            else:
                participants = self.users

            time_start = time.time()
            self.weighted_execute_update_with_affine_dataset(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)

            avg_state_dict, avg_loss = self.weighted_aggregate(participants)
            print(f'Averaged train loss:{avg_loss:.4f}')
            self.update_global_model(avg_state_dict)

            overall_acc_clean, loss_clean, per_class_acc_clean, overall_acc_poison, loss_poison, per_class_acc_poison = self.test_global_model()
            self.global_model_records[len(self.global_model_records)] = {
                'train_loss': avg_loss,
                'overall_acc_clean': overall_acc_clean,
                'loss_clean': loss_clean,
                'per_class_acc_clean': per_class_acc_clean,
                'overall_acc_poison': overall_acc_poison,
                'loss_poison': loss_poison,
                'per_class_acc_poison': per_class_acc_poison
            }

        self.save()

        root = os.path.join(self.save_path, f'{self.args.dataset}-{self.args.env}-r{self.args.communication_round}')
        os.makedirs(root, exist_ok=True)
        accuracy_filename = f'results-train.pkl'
        accuracy_filepath = os.path.join(root, accuracy_filename)
        with open(accuracy_filepath, 'wb') as f:
            pickle.dump(self.global_model_records, f)
        
        print(f'{get_time()} Global Update Finished Using {time_cnt}')


    def weighted_execute_update_with_affine_dataset(self, participants: list):
        for client in participants:
            self.client_instances[client].local_update_with_affine_dataset(self.local_epoch, self.global_model, self.lr, self.weight_decay, self.momentum)


    def save_affine_dataset(self):
        if self.args.env_path is None:
            root = '../env/{}/{}-{}-scale{}'.format(self.args.affine_path, self.args.dataset, self.args.env, self.args.scale)
        else:
            root = '{}/{}/{}-{}-scale{}'.format(self.args.env_path, self.args.affine_path, self.args.dataset, self.args.env, self.args.scale)
        if not os.path.exists(root):
            os.makedirs(root)
        train_root = '{}/{}'.format(root, 'train')

        if not os.path.exists(train_root):
            os.makedirs(train_root)

        train_path = 'train.pt'
        train_dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        for uid in self.users:
            train_dataset['users'].append(uid)
            # print(ys[i])
            current_client = self.client_instances[uid]
            train_dataset['user_data'][uid] = {
                'x': current_client.image_syn_train,
                'y': current_client.label_syn_train}
            train_dataset['num_samples'].append(len(current_client.label_syn_train))
        print(f"Clients: {train_dataset['users']}")
        print(f"Affine dataset result: {train_dataset['num_samples']}")
        train_save_path = '{}/{}'.format(train_root, train_path)
        print('save to {}'.format(train_save_path))
        torch.save(train_dataset, train_save_path)


class FedQuickDropClient(FedAvgClient):
    def __init__(self, args, id, device, trainset):
        super().__init__(args, id, device, trainset)
        
        self.image_syn_train = None
        self.label_syn_train = None

    def load_affine_data(self, x_tensor, y_tensor):
        self.image_syn_train = x_tensor
        self.label_syn_train = y_tensor

    def local_update_with_affine_dataset(self, local_epoch, model, lr, weight_decay, momentum):
        # print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        self.local_model = copy.deepcopy(model)

        ''' indicate the desired dataset '''
        dst_train = self.trainset
        ''' organize the dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(self.args.num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.args.device)

        ''' extraction logics '''
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        ''' quantity of distilled images '''
        ipcs = [math.ceil(len(indices_class[c]) * self.args.scale) for c in range(self.args.num_classes)]

        ''' initialize the synthetic data '''
        # print(f'\t{get_time()} initialize {int(sum(ipcs))} synthetic data')
        image_syn = torch.randn(size=(sum(ipcs), self.args.channel, self.args.im_size[0], self.args.im_size[1]), dtype=torch.float, requires_grad=True, device=self.args.device)
        label_syn = np.concatenate([np.ones(ipcs[c]) * c for c in range(self.args.num_classes) if ipcs[c] != 0])
        label_syn = torch.tensor(label_syn, dtype=torch.long, requires_grad=False, device=self.args.device).view(-1)
        # print(label_syn)

        if self.args.init == 'real':
            # print('\tinitialize synthetic data from random real images')
            # override the random noise
            for c in range(self.args.num_classes): image_syn.data[sum(ipcs[:c]):sum(ipcs[:c + 1])] = get_images(c, ipcs[c]).detach().data
        else:
            # print('\tinitialize synthetic data from random noise')
            pass

        ''' prepare for training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.args.lr_img, momentum=momentum, weight_decay=weight_decay)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        net_parameters = list(self.local_model.parameters())
        optimizer_net = torch.optim.SGD(self.local_model.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        self.args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


        epoch_loss = []
        ''' train synthetic data '''
        for ol in range(local_epoch):
            ''' freeze the running mu and sigma for BatchNorm layers '''
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the training with BatchNorm layers easier.
            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in self.local_model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(self.args.num_classes)], dim=0)
                self.local_model.train()  # for updating the mu, sigma of BatchNorm
                output_real = self.local_model(img_real)  # get running mu, sigma
                for module in self.local_model.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        module.eval()  # fix mu and sigma of every BatchNorm layer
            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(self.args.device)
            accumulated_grads = [torch.zeros_like(param) for param in net_parameters]
            for c in range(self.args.num_classes):  # c == il
                if ipcs[c] == 0:
                    continue
                if ipcs[c] == 1 and len(indices_class[c]) == 1:
                    continue
                img_real = get_images(c, self.args.batch_real if self.args.batch_real >= ipcs[c] else ipcs[c])
                lab_real = torch.ones((img_real.shape[0],), device=self.args.device, dtype=torch.long) * c
                img_syn = image_syn[sum(ipcs[:c]): sum(ipcs[:c + 1])].reshape((ipcs[c], self.args.channel, self.args.im_size[0], self.args.im_size[1]))
                lab_syn = torch.ones((ipcs[c],), device=self.args.device, dtype=torch.long) * c

                if self.args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, self.args.dsa_strategy, seed=seed, param=self.args.dsa_param)
                    img_syn = DiffAugment(img_syn, self.args.dsa_strategy, seed=seed, param=self.args.dsa_param)


                output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                output_syn = self.local_model(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True, retain_graph=True)
                if self.args.directly_update:
                    accumulated_grads = [accum_grad + g for accum_grad, g in zip(accumulated_grads, gw_real)]
                loss += match_loss(gw_syn, gw_real, self.args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                label_syn.detach())  # avoid any unaware modification
            self.image_syn_train = image_syn_train
            self.label_syn_train = label_syn_train

            if self.args.directly_update:
                with torch.no_grad():  # We do not want to track operations here
                    for param, grad in zip(self.local_model.parameters(), accumulated_grads):
                        # grad += self.args.weight_decay * param # Apply weight decay
                        param -= self.args.lr_net * grad  # Apply gradient descent update
            else:
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, batch_size=self.args.batch_size, shuffle=True,
                                         num_workers=self.args.num_workers)
                for il in range(local_epoch):
                    epoch('train', trainloader, self.local_model, optimizer_net, criterion, self.args, aug=True if self.args.dsa else False)
            
            epoch_loss.append(loss.item())
        self.loss_records.append(sum(epoch_loss) / len(epoch_loss))
        # print('{} {}'.format(get_time(), 'Client {} completes update'.format(self.id)))


class FedQuickDrop_unlearn(FedQuickDropServer):
    def __init__(self, args, trained_server: FedQuickDropServer):
        super().__init__(args)
        self.trained_server = trained_server
        self.unlearn_model = copy.deepcopy(self.trained_server.global_model)
        self.client_instances = self.trained_server.client_instances  


    def unlearn_process(self, client_idx):
        client = self.client_instances[client_idx]
        syn_dataset = TensorDataset(client.image_syn_train, client.label_syn_train)
        new_dataloader = DataLoader(syn_dataset, batch_size=self.args.unlearn_bs, shuffle=True)

        recall_net = copy.deepcopy(self.unlearn_model)

        optimizer = torch.optim.SGD(self.unlearn_model.parameters(), lr=self.args.forgetting_lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.forgetting_epoch):
            epoch_loss = 0.0
            for batch_data, batch_labels in new_dataloader:
                batch_data = batch_data.to(self.args.device)
                batch_labels = batch_labels.to(self.args.device)
                optimizer.zero_grad()

                output = self.unlearn_model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

            self.unlearn_model.load_state_dict(self.compute_sga_grad(recall_net, self.unlearn_model))

            # test on testset
            overall_acc_clean, loss_clean, per_class_acc_clean = test_loop(self.device, self.clean_test_loader, self.unlearn_model, self._loss_fc, self.num_classes)
            overall_acc_poison, loss_poison, per_class_acc_poison = test_loop(self.device, self.poisoned_test_loader, self.unlearn_model, self._loss_fc, self.num_classes)
            print(f"Clean Accuracy for global model: {overall_acc_clean:.4f}, loss: {loss_clean:.4f}")
            print(f"Backdoor Accuracy for global model: {overall_acc_poison:.4f}, loss: {loss_poison:.4f}")

            self.global_model_records[len(self.global_model_records)] = {
                'unlearning_loss': 0,
                'overall_acc_clean': overall_acc_clean,
                'loss_clean': loss_clean,
                'per_class_acc_clean': per_class_acc_clean,
                'overall_acc_poison': overall_acc_poison,
                'loss_poison': loss_poison,
                'per_class_acc_poison': per_class_acc_poison
            }

        self.save()
        root = os.path.join(self.save_path, f'{self.args.dataset}-{self.args.env}-r{self.args.communication_round}')
        os.makedirs(root, exist_ok=True)
        accuracy_filename = f'results-unlearning.pkl'
        accuracy_filepath = os.path.join(root, accuracy_filename)
        with open(accuracy_filepath, 'wb') as f:
            pickle.dump(self.global_model_records, f)

        print(f"Unlearning process completed for client {client_idx}.")
        self.trained_server.global_model = self.unlearn_model
        return self.trained_server


    def compute_sga_grad(self, recall_net, curr_net):
        final_state_dict = copy.deepcopy(recall_net.state_dict())
        pres_state_dict = recall_net.state_dict()
        curr_state_dict = curr_net.state_dict()
        for k in final_state_dict.keys():
            final_state_dict[k] += (pres_state_dict[k] - curr_state_dict[k])
        return final_state_dict


class FedQuickDrop_recover(FedQuickDrop_unlearn):
    def __init__(self, args, unlearn: FedQuickDrop_unlearn):
        super().__init__(args, unlearn.trained_server)

        self.communication_round_recover = args.communication_round_recover
        self.server_unlearn = unlearn  
        self.global_model = self.trained_server.global_model

        self.client_instances = {
            uid: FedQuickDropClient(args, uid, self.device, self.user_trainsets[uid])
            for uid in self.users
        }

        print(f"Recover Server: Available clients = {list(self.client_instances.keys())}")

        # load affine data
        affine_data_path = os.path.join(
            '../env/quickdrop-affine',
            f"{self.args.dataset}-{self.args.env}-scale{self.args.scale}",
            'train',
            'train.pt'
        )
        affine_data = torch.load(affine_data_path)

        for uid in self.users:
            x = affine_data["user_data"][uid]["x"]
            y = affine_data["user_data"][uid]["y"]
            self.client_instances[uid].load_affine_data(x, y)


    def recover_global_update(self, recover_epoch, client_idx, n=-1):
        time_cnt = 0
        for r in range(self.communication_round_recover):
            print(f'{get_time()} Global Recover Round {r}')

            available_clients = list(self.client_instances.keys())  

            if client_idx in available_clients:
                available_clients.remove(client_idx)
                print(f"Removed client {client_idx} from available clients.")


            if not available_clients:
                print("No available clients for recovery training.")
                return

            if n != -1:
                participants = random.sample(available_clients, min(n, len(available_clients)))
            else:
                participants = available_clients


            if not participants:
                print("No available clients for recovery training.")
                return


            print(f"Participants in this round: {participants}")
            time_start = time.time()
            self.execute_recover_update(recover_epoch, participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)


            avg_state_dict, avg_loss = self.weighted_aggregate(participants)
            print(f'Averaged recover train loss: {avg_loss:.4f}')
            self.update_global_model(avg_state_dict)
            overall_acc_clean, loss_clean, per_class_acc_clean, overall_acc_poison, loss_poison, per_class_acc_poison = self.test_global_model()
            self.global_model_records[len(self.global_model_records)] = {
                'train_loss': avg_loss,
                'overall_acc_clean': overall_acc_clean,
                'loss_clean': loss_clean,
                'per_class_acc_clean': per_class_acc_clean,
                'overall_acc_poison': overall_acc_poison,
                'loss_poison': loss_poison,
                'per_class_acc_poison': per_class_acc_poison
            }

        self.save()
        root = os.path.join(self.save_path, f'{self.args.dataset}-{self.args.env}-r{self.args.communication_round}')
        os.makedirs(root, exist_ok=True)
        accuracy_filename = f'results-recovery.pkl'
        accuracy_filepath = os.path.join(root, accuracy_filename)
        with open(accuracy_filepath, 'wb') as f:
            pickle.dump(self.global_model_records, f)

        print(f'{get_time()} Global Recovery Finished Using {time_cnt}')


# define args
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
parser.add_argument('--client_idx', type=str, default='f_00003', help='backdoor client')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--env_path', type=str, default='../backdoor', help='environment path')
parser.add_argument('--strategy', type=str, default='dilichlet', help='strategy')
parser.add_argument('--env', type=str, default='seed0-u10-alpha0.1', help='FL env')
parser.add_argument('--communication_round', type=int, default=50, help='FL communication round, 100')
# Training hyperparameters:
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for normal global update')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Learning rate decay if applicable')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
parser.add_argument('--save_path', type=str, default='../save/quickdrop')

parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--unlearn_bs', type=int, default=32, help='Batch size of unlearning')
parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
parser.add_argument('--directly_update', type=bool, default=False, help='True: update local model via synthetic loss/False: recalculate loss on synthetic dataset')
parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for updating synthetic images')
parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--affine_path', type=str, default='quickdrop-affine', help='path to save affine results')
parser.add_argument('--scale', type=float, default=0.01, help='For each class: #dc images = #original images * ratio')
parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
# Forgetting parameters
parser.add_argument('--forgetting_epoch', type=int, default=5, help='FL forgetting epoch')
parser.add_argument('--forgetting_lr', type=float, default=0.001, help='Forgetting learning rate')
parser.add_argument('--recover_epoch', type=int, default=5, help='FL recover epoch')
parser.add_argument('--communication_round_recover', type=int, default=10, help='FL communication round, 200')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory')
parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args("")
args.dsa_param = ParamDiffAug()
args.dsa = True if args.method == 'DSA' else False

setup_seed(args.seed)

# training
server = FedQuickDropServer(args)
server.weighted_global_update(n=-1)

torch.save(server.global_model,
            '../check_point/Quickdrop(FL)-{}-{}-r{}.pth'.format(args.dataset, args.env, args.communication_round))

server.save_affine_dataset()

# unlearning
# use affine dataset
unlearn = FedQuickDrop_unlearn(args, server)
unlearn.unlearn_process(args.client_idx)
server.global_model = unlearn.unlearn_model

# recovery 
# use affine dataset
recover_server = FedQuickDrop_recover(args, unlearn)
recover_server.recover_global_update(args.recover_epoch, args.client_idx, n=-1)

torch.save(recover_server.global_model,
            '../check_point/Quickdrop(Recovery)-{}-{}-fe{}-rc{}.pth'.format(args.dataset, args.env, args.forgetting_epoch, args.communication_round_recover))
