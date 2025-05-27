#%%
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
from diffusers import AutoencoderKL
from sklearn.decomposition import PCA
from geomloss import SamplesLoss
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


def apply_principal_direction(z, principal_dirs, dir_idx=0, alpha=1.0):
    direction = principal_dirs[dir_idx].view_as(z[0])  # [C, H, W]
    return z + alpha * direction  # [N, C, H, W]

def interpolate_models(model_a, model_b, alpha=0.5):
    model_interp = copy.deepcopy(model_a)
    with torch.no_grad():
        for param_a, param_b, param_interp in zip(model_a.parameters(), model_b.parameters(), model_interp.parameters()):
            param_interp.data.copy_(alpha * param_a.data + (1 - alpha) * param_b.data)
    return model_interp


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

        self._loss_fc = nn.CrossEntropyLoss()
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


class FuGuardClient(FedAvgClient):
    def __init__(self, args, id, device, trainset):
        super().__init__(args, id, device, trainset)

    def local_update(self, epoch, model, lr, weight_decay, momentum):
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


class FuGuardServer_unlearn(FedAvgServer):
    def __init__(self, args, trained_server: FedAvgServer, load_gan=True):
        super().__init__(args)
        self.trained_server = trained_server
        self.unlearn_model = copy.deepcopy(self.trained_server.global_model)
        self.client_instances = self.trained_server.client_instances  

        self.gan_model = None

    def load_pretrained_gan(self):
        if self.gan_model is None:
            try:
                print("Loading pre-trained GAN model...")
                self.gan_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
                self.gan_model.to(self.args.device)
                print("GAN model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load GAN model: {e}")


    def unlearn_process(self, client_idx):
        self.load_pretrained_gan()
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
        
        client = self.client_instances[client_idx]

        dataset_len = len(client.train_dataloader.dataset)
        samples_num = int(self.args.samples_scale * dataset_len)
        print(f"sampling {samples_num} data")
        indices = random.sample(range(dataset_len), samples_num)

        sampled_data = [client.train_dataloader.dataset[i] for i in indices]
        some_images = torch.stack([image.to(self.args.device) for image, _ in sampled_data])

        sampled_labels = torch.tensor([label for _, label in sampled_data], dtype=torch.long, device=self.args.device)
        sampled_dataset = TensorDataset(some_images, sampled_labels)

        sampled_dataloader = DataLoader(sampled_dataset, batch_size=self.args.gen_bs, shuffle=False)

        z_B_list = []
        with torch.no_grad():
            for images, _ in sampled_dataloader:
                latent = self.gan_model.encode(images).latent_dist
                z_B_list.append(latent.mean)

        z_B = torch.cat(z_B_list, dim=0)

        # Step 1: flatten latent
        z_B_flat = z_B.view(z_B.size(0), -1)  # [N, D]

        # Step 2: PCA
        pca = PCA(n_components=5)
        pca.fit(z_B_flat.cpu().numpy())
        principal_dirs = torch.tensor(pca.components_, device=z_B.device)  # [5, D]

        z_new = apply_principal_direction(z_B, principal_dirs, dir_idx=1, alpha=1.0)

        # Step 4: decode
        new_data = self.gan_model.decode(z_new).sample
        new_data = new_data.clamp(0, 1)

        print(f"Generated new data with shape: {new_data.shape}")

        del self.gan_model
        torch.cuda.empty_cache()

        # new dataset
        new_dataset = TensorDataset(new_data, sampled_labels)
        new_dataloader = DataLoader(new_dataset, batch_size=self.args.unlearn_bs, shuffle=True)

        old_model = copy.deepcopy(self.trained_server.global_model)
        old_model.eval()

        with torch.no_grad():
            z_old_all = []
            for batch_data, _ in new_dataloader:
                batch_data = batch_data.to(self.args.device)
                z_old = old_model.embed(batch_data)
                z_old_all.append(z_old)
            z_old_all = torch.cat(z_old_all, dim=0)

        # SGA
        optimizer = torch.optim.SGD(self.unlearn_model.parameters(), lr=self.args.forgetting_lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.forgetting_epoch):
            epoch_loss = 0.0
            for i, (batch_data, batch_labels) in enumerate(new_dataloader):
                batch_data = batch_data.to(self.args.device)
                batch_labels = batch_labels.to(self.args.device)
                optimizer.zero_grad()

                outputs = self.unlearn_model(batch_data)
                z_new = self.unlearn_model.embed(batch_data)
                z_old = z_old_all[i * batch_data.size(0):(i + 1) * batch_data.size(0)]
                # OT loss
                ot_loss = sinkhorn_loss(z_new, z_old)

                # CrossEntropy loss
                ce_loss = torch.nn.functional.cross_entropy(outputs, batch_labels)

                # total loss
                total_loss = -ce_loss + self.args.ot_lambda * ot_loss
                total_loss.backward()
                # if self.args.clip_grad > 0:
                #     torch.nn.utils.clip_grad_norm_(self.unlearn_model.parameters(), self.args.clip_grad)
                optimizer.step()

                epoch_loss += total_loss.item()


            unlearning_loss = -epoch_loss / len(new_dataloader)
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
        interpolated_model = interpolate_models(old_model, self.unlearn_model, alpha=args.alpha)
        self.trained_server.global_model = interpolated_model
        return self.trained_server


class FUGenServer_recover(FuGuardServer_unlearn):
    def __init__(self, args, unlearn: FuGuardServer_unlearn):
        super().__init__(args, unlearn.trained_server, load_gan=False)

        self.communication_round_recover = args.communication_round_recover
        self.server_unlearn = unlearn  
        self.global_model = self.trained_server.global_model

        self.client_instances = {
            uid: FuGuardClient(args, uid, self.device, self.user_trainsets[uid])
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
parser.add_argument('--model', type=str, default='ConvNet', help='MLP, ConvNet')
parser.add_argument('--env_path', type=str, default='../backdoor', help='environment path')
parser.add_argument('--strategy', type=str, default='dilichlet', help='strategy')
parser.add_argument('--env', type=str, default='seed0-u10-alpha0.1', help='FL env')
parser.add_argument('--communication_round', type=int, default=50, help='FL communication round, 100')
# Training hyperparameters:
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for normal global update')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Learning rate decay if applicable')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--local_epoch', type=int, default=5, help='FL local epoch')
parser.add_argument('--save_path', type=str, default='../save/fuguard')

parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
# parser.add_argument('--clip_grad', type=int, default=35, help=' ')
parser.add_argument('--samples_scale', type=int, default=0.1, help='number of sampling for generation')
parser.add_argument('--gen_bs', type=int, default=64, help='Batch size of generating')
parser.add_argument('--unlearn_bs', type=int, default=64, help='Batch size of unlearning')
parser.add_argument('--ot_lambda', type=float, default=0.1, help='op unlearning')
parser.add_argument('--alpha', type=float, default=0.5, help='')

parser.add_argument('--forgetting_epoch', type=int, default=5, help='FL forgetting epoch')
parser.add_argument('--forgetting_lr', type=float, default=0.001, help='Forgetting learning rate')   # svhn 0.001  CIFAR10 0.003
parser.add_argument('--recover_epoch', type=int, default=5, help='FL recover epoch')
parser.add_argument('--communication_round_recover', type=int, default=10, help='FL communication round, 200')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--pin_memory', type=bool, default=False, help='Pin memory')
parser.add_argument('--persistent_workers', type=bool, default=False, help='Persistent workers')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args("")

setup_seed(args.seed)

server = FedAvgServer(args)
server.weighted_global_update(n=-1)

torch.save(server.global_model,
            '../check_point/FuGuard(FL)-{}-{}-r{}.pth'.format(args.dataset, args.env, args.communication_round))

# unlearning
unlearn = FuGuardServer_unlearn(args, server)
unlearn.unlearn_process(args.client_idx)
server.global_model = unlearn.unlearn_model

# recovery
recover_server = FUGenServer_recover(args, unlearn)
recover_server.recover_global_update(args.recover_epoch, args.client_idx, n=-1)

torch.save(recover_server.global_model,
            '../check_point/FuGuard(Recovery)-{}-{}-fe{}-rc{}.pth'.format(args.dataset, args.env, args.forgetting_epoch, args.communication_round_recover))
