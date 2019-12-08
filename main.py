import torch
from torchvision import datasets, transforms
import sampler as custom_sampler
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
from utils import *
import arguments

from torch.optim import RMSprop
import copy

def cifar_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])


class ReplayBuffer():
    def __init__(self):
        self.rb_data = []

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def insert_episode(self, episode):
        self.rb_data.append(episode)

    @property
    def episodes_in_buffer(self):
        return len(self.rb_data)

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    #TODO add automatically formating of the output data
    def __getitem__(self, item):
        print("in __getitem__")
        print(item)
        if type(item) == slice:
            i_start, i_end, _ = item
            return self.rb_data[i_start: i_end]
        elif type(item) == list:
            output = []
            for i in item:
                output.append(self.rb_data[i])
            return output


class RLLearner():
    def __init__(self, q_net, rlc, args):
        self.args = args
        self.q_net = q_net
        self.rlc = rlc

        self.params = list(q_net.parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_q_net = copy.deepcopy(q_net)

        # self.log_stats_t = -self.args.learner_log_interval - 1
    
    def train(self, episode_batch):
        # Get the relevant quantities
        #TODO calculate the reward based on the observation
        rewards = episode_batch["reward"][:, :-1]
        actions = episode_batch["actions"][:, :-1]
        # terminated = batch["terminated"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = episode_batch["avail_actions"]

        # Calculate estimated Q-Values
        agent_outs = []
        self.rlc.init_hidden(episode_batch.batch_size)
        for t in range(episode_batch.max_seq_length):
            agent_out = self.q_net.forward(episode_batch, t=t)
            # print("agent_out size")
            # print(agent_outs.size())
            agent_outs.append(agent_out)
        agent_outs = torch.stack(agent_outs, dim=1)  # Concat over time

        print("MAC_OUT size:")
        print(agent_outs.size())
        print(actions.size())

        # Pick the Q-Values for the actions taken by each agent
        #TODO check the following line of code
        chosen_action_qvals = torch.gather(agent_outs[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # print(th.gather(mac_out[:, :-1], dim=3, index=actions).size())
        print(chosen_action_qvals.size())

        # Calculate the Q-Values necessary for the target
        target_agent_outs = []
        self.target_q_net.init_hidden(episode_batch.batch_size)
        for t in range(episode_batch.max_seq_length):
            target_agent_out = self.target_q_net.forward(episode_batch, t=t)
            target_agent_outs.append(target_agent_out)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_agent_outs = torch.stack(target_agent_outs[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_agent_outs[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            agent_outs[avail_actions == 0] = -9999999
            cur_max_actions = agent_outs[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_agent_outs, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_agent_outs.max(dim=3)[0]

        
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).sum() / mask.sum()


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_q_net.load_state(self.q_net)
        print("Updated target network")

    def cuda(self):
        self.q_net.cuda()
        self.target_q_net.cuda()

    def save_models(self, path):
        self.q_net.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.q_net.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_q_net.load_models(path)
    
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class RLController():
    def __init__(self, args, q_net, train_dataset):
        self.rnn_hidden_state = None
        self.q_net = q_net
        self.train_dataset = train_dataset
        self.args = args

    def init_hidden(self):
        self.rnn_hidden_state = self.q_net.init_hidden()

    def select_actions(self, unlabeled_indices):
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)

        unlabeled_dataloader_rl = data.DataLoader(self.train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        all_q = []
        all_indices = []
        for images, _, indices in unlabeled_dataloader_rl:
            if self.args.cuda:
                images = images.cuda()

            with torch.no_grad():
                #TODO add other features to the network
                q, _ = self.q_net(images, self.rnn_hidden_state)

            
            all_q.extend(q)
            all_indices.extend(indices)

        all_q = torch.stack(all_q)
        all_q = all_q.view(-1)
        print("@@@@@@@@@@@@@@@@")
        print(all_q)
        print(all_q.size())
        
        # Action selection
        sampler_on_q = custom_sampler.TopSampler(self.args.budget)
        sampled_indices = sampler_on_q.sample(all_q, all_indices)
        print("sampled_indices:")
        print(sampled_indices)
        best_q = None
        
        temp_sampler = data.sampler.SubsetRandomSampler(list(sampled_indices))
        best_q_image_dataloader = data.DataLoader(self.train_dataset, sampler=temp_sampler, 
                batch_size=self.args.batch_size, drop_last=False)
        for images, _, indices in best_q_image_dataloader:
            print("image: %s" % str(images))
            if self.args.cuda:
                images = images.cuda()
            best_q, self.rnn_hidden_state = self.q_net(images, self.rnn_hidden_state)
            print("curr q: %s" % q)
        
        return sampled_indices, best_q



def run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc, test_mode=False):
    sampler = data.sampler.SubsetRandomSampler(initial_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)

    rlc.init_hidden()

    splits = [args.initial_budget,
        (args.initial_budget+args.budget), 
        (args.initial_budget+args.budget*2), 
        (args.initial_budget+args.budget*3), 
        (args.initial_budget+args.budget*4), 
        (args.initial_budget+args.budget*5)]

    episode_info = []

    current_indices = list(initial_indices)
    accuracies = []
    
    for t, split in enumerate(splits):
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)

        print("RNN hidden input: %s" % str(rlc.rnn_hidden_state))

        task_model = model.FCNet(num_classes=args.num_classes)
        if args.dataset == "mnist":
            vae = model.VAE(args.latent_dim, nc=1)
        elif args.dataset == "ring":
            vae = model.VAE(args.latent_dim, nc=2)
        else:
            vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim, multi_class=args.is_multi_class)


        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        if args.sampling_method == "adversary" or args.sampling_method == "adversary_1c":
            # train the models on the current data
            acc, vae, discriminator = solver.train(querry_dataloader,
                                                task_model, 
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader)
        else:
            # train the models on the current data
            acc, vae, discriminator = solver.train_without_adv_vae(querry_dataloader,
                                                task_model, 
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader)

        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        accuracies.append(acc)

        pre_transition_data = {'state': unlabeled_indices, 'accurcy': acc, 't':t}
        episode_info.append(pre_transition_data)

        sampled_indices, best_q = rlc.select_actions(unlabeled_indices)
        
        print("len of current_indices before adding sample indices: %s" % len(current_indices))
        current_indices = list(current_indices) + list(sampled_indices)
        print("len of current_indices after adding sample indices: %s" % len(current_indices))
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=False)

    return accuracies, episode_info

def main(args):
    if args.dataset == "ring":
        print("Using Ring dataset...")
        test_dataloader = data.DataLoader(
            Ring(args.data_path, transform=simple_data_transformer(), return_idx=False, testset=True),
            batch_size=args.batch_size, drop_last=False
        )

        train_dataset = Ring(args.data_path, simple_data_transformer())
        print(len(train_dataset))
        args.num_images = 2500
        args.budget = 1
        args.initial_budget = 1
        args.num_classes = 5 
    
    elif args.dataset == 'mnist':
        test_dataloader = data.DataLoader(
                datasets.MNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = MNIST(args.data_path)
        print(len(train_dataset))
        args.num_images = 6000
        args.budget = 300
        args.initial_budget = 300
        args.num_classes = 10

    elif args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 5000
        args.budget = 250
        args.initial_budget = 500
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    random.seed("csc2547")

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    
            
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    q_net = model.RnnNet(2, 3)
    if args.cuda:
        q_net.cuda()

    
    replay_buffer = ReplayBuffer()
    rlc = RLController(args, q_net, train_dataset)
    rl_learner = RLLearner(q_net, None, args)


    num_runs = 2
    batch_size = 4
    for i in range(num_runs):
        episode_info = run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc)

        print("!!!!!!!!!!!!!!!!!")
        print(episode_info)

        replay_buffer.insert_episode(episode_info)

        if replay_buffer.can_sample(batch_size):
            episode_sample = replay_buffer.sample(batch_size)
            if args.cuda:
                episode_sample.cuda()

            rl_learner.train(episode_sample)

        eval_interval = 20
        if i % eval_interval == 0:
            run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc)
        #TODO save RL model here
    



    # torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

