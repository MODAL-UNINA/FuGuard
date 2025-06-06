# This file includes portions of code adapted from https://github.com/sacs-epfl/quickdrop
# Credit to the original authors. Modifications have been made to fit the needs of this project.

import argparse
import copy
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    if root_path is None:
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

    def _init_model(self):
        return init_model(self.model, self.channel, self.num_classes, self.im_size).to(self.device)


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
            self.client_instances[client].local_update(self.local_epoch, self.global_model, self.lr, self.weight_decay, self.momentum)

    def execute_recover_update(self, recover_epoch, participants: list):
        for client in participants:
            self.client_instances[client].local_update(recover_epoch, self.global_model, self.lr, self.weight_decay, self.momentum)

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
            avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])
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
    def local_update(self, epoch, model, lr, weight_decay, momentum):
        # print('{} {}'.format(get_time(), 'Client {} updates'.format(self.id)))
        self.local_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        epoch_loss = []
        for i in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
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


class FUServer(FedAvgServer):
    def __init__(self, args):
        super().__init__(args)
        self.last_round_party_models = {}

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
            self.execute_update(participants)
            time_end = time.time()
            time_cnt += (time_end - time_start)

            avg_state_dict, avg_loss = self.weighted_aggregate(participants)
            print(f'Averaged train loss:{avg_loss:.4f}')
            self.update_global_model(avg_state_dict)

            for client in participants:
                self.last_round_party_models[client] = copy.deepcopy(
                    self.client_instances[client].local_model.state_dict()
                )

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


class FUClient(FedAvgClient):
    def __init__(self, args, id, device, trainset):
        super().__init__(args, id, device, trainset)

    def local_update(self, epoch, model, lr, weight_decay, momentum):
        self.local_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.local_model.train()
        epoch_loss = []
        for i in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
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


class FUServer_unlearn(FUServer):
    def __init__(self, args, trained_server: FUServer):
        super().__init__(args)
        self.trained_server = trained_server
        self.fedavg_model = copy.deepcopy(self.trained_server.global_model)
        self.client_instances = self.trained_server.client_instances
        self.last_round_party_models = self.trained_server.last_round_party_models
        self.unlearn_model = None

    def get_distance(self, model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance


    def compute_reference_model(self, client_idx):
        client_model = copy.deepcopy(self.fedavg_model)
        client_model.load_state_dict(self.last_round_party_models[client_idx])

        #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
        num_parties = len(self.client_instances)
        model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(self.fedavg_model.parameters()) \
                                    - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(client_model.parameters())

        #compute threshold
        model_ref = self._init_model()
        nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

        dist_ref_random_lst = []
        for _ in range(10):
            dist_ref_random_lst.append(self.get_distance(model_ref, self._init_model()))   

        dist_ref_random_lst = [dist.cpu().item() for dist in dist_ref_random_lst]
        threshold = np.mean(dist_ref_random_lst) / 3

        return client_model, model_ref, threshold


    def unlearn_process(self, client_idx):
        client = self.client_instances[client_idx]
        client_model, model_ref, threshold = self.compute_reference_model(client_idx)
        self.unlearn_model = copy.deepcopy(model_ref)

        optimizer = torch.optim.SGD(self.unlearn_model.parameters(), lr=self.args.forgetting_lr)
        criterion = torch.nn.CrossEntropyLoss()
        flag = False

        for epoch in range(self.args.forgetting_epoch):
            print('------------', epoch)
            if flag:
                break
            epoch_loss = 0.0
            for batch_data, batch_labels in client.train_dataloader:
                batch_data = batch_data.to(self.args.device)
                batch_labels = batch_labels.to(self.args.device)
                optimizer.zero_grad()
                
                output = self.unlearn_model(batch_data)
                loss = -criterion(output, batch_labels)
                loss.backward()
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.unlearn_model.parameters(), self.args.clip_grad)
                optimizer.step()
                
                epoch_loss += loss.item()

                with torch.no_grad():
                    distance = self.get_distance(self.unlearn_model, model_ref)
                    if distance > threshold:
                        dist_vec = nn.utils.parameters_to_vector(self.unlearn_model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                        dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(threshold)
                        proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        nn.utils.vector_to_parameters(proj_vec, self.unlearn_model.parameters())
                        distance = self.get_distance(self.unlearn_model, model_ref)

                distance_ref_target = self.get_distance(self.unlearn_model, client_model)
                print('Distance from the unlearned model to client model:', distance_ref_target.item())

                if distance_ref_target > self.args.distance_threshold:
                    flag = True
                    break

            unlearning_loss = -epoch_loss / len(client.train_dataloader)
            print(f"Epoch {epoch}: Unlearning loss = {unlearning_loss}")

            # test on testset
            overall_acc_clean, loss_clean, per_class_acc_clean = test_loop(self.device, self.clean_test_loader, self.unlearn_model, self._loss_fc, self.num_classes)
            overall_acc_poison, loss_poison, per_class_acc_poison = test_loop(self.device, self.poisoned_test_loader, self.unlearn_model, self._loss_fc, self.num_classes)
            print(f"Clean Accuracy for global model: {overall_acc_clean:.4f}, loss: {loss_clean:.4f}")
            print(f"Backdoor Accuracy for global model: {overall_acc_poison:.4f}, loss: {loss_poison:.4f}")

            self.global_model_records[len(self.global_model_records)] = {
                'unlearning_loss': unlearning_loss,
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


class FUServer_recover(FUServer_unlearn):
    def __init__(self, args, unlearn: FUServer_unlearn):
        super().__init__(args, unlearn.trained_server)

        self.communication_round_recover = args.communication_round_recover
        self.server_unlearn = unlearn  
        self.global_model = self.trained_server.global_model

        self.client_instances = {
            uid: FUClient(args, uid, self.device, self.user_trainsets[uid])
            for uid in self.users
        }

        print(f"Recover Server: Available clients = {list(self.client_instances.keys())}")


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

            avg_state_dict, avg_loss = self.weighted_aggregate(participants)  # 聚合更新
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
# Device: cpu/gpu/gpu:# if you want to indicate a specific running device
parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
parser.add_argument('--client_idx', type=str, default='f_00003', help='backdoor client')
parser.add_argument('--model', type=str, default='ConvNet', help='MLP, ConvNet')
parser.add_argument('--env_path', type=str, default='../backdoor', help='environment path')
parser.add_argument('--strategy', type=str, default='dilichlet', help='strategy')
parser.add_argument('--env', type=str, default='seed0-u10-alpha0.1', help='FL env')
parser.add_argument('--communication_round', type=int, default=50, help='FL communication round, 100')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for normal global update')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Learning rate decay if applicable')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
parser.add_argument('--save_path', type=str, default='../save/fu')

parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--clip_grad', type=int, default=5, help=' ')
parser.add_argument('--unlearn_bs', type=int, default=256, help='Batch size of unlearning')
parser.add_argument('--distance_threshold', type=float, default=2.2, help='')

parser.add_argument('--forgetting_epoch', type=int, default=5, help='FL forgetting epoch')
parser.add_argument('--forgetting_lr', type=float, default=0.001, help='Forgetting learning rate')
parser.add_argument('--recover_epoch', type=int, default=5, help='FL recover epoch')
parser.add_argument('--communication_round_recover', type=int, default=10, help='')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory')
parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args("")

setup_seed(args.seed)

# training
server = FUServer(args)
server.weighted_global_update(n=-1)
torch.save(server.global_model,
            '../check_point/FU(FL)-{}-{}-r{}.pth'.format(args.dataset, args.env, args.communication_round))

# unlearning
unlearn = FUServer_unlearn(args, server)
unlearn.unlearn_process(args.client_idx)
server.global_model = unlearn.unlearn_model

# recovery
recover_server = FUServer_recover(args, unlearn)
recover_server.recover_global_update(args.recover_epoch, args.client_idx, n=-1)

torch.save(recover_server.global_model,
            '../check_point/FU(Recovery)-{}-{}-fe{}-rc{}.pth'.format(args.dataset, args.env, args.forgetting_epoch, args.communication_round_recover))
