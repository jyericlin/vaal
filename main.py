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
from torch.distributions import Categorical

import coloredlogs, logging
coloredlogs.install(level='INFO')



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

    def get_formatted_data(self, preproc_data, data_type, format_as_tensor=True):
        output_list = []
        for episode in preproc_data:
            episode_data = [step[data_type] for step in episode]
            if format_as_tensor:
                output_list.append(torch.stack(episode_data))
            else:
                output_list.append(episode_data)
        if format_as_tensor:
            output = torch.stack(output_list)
        else:
            output = output_list
        # print(output)
        # if type(output) == torch.Tensor:
        #     print(output.size())
        return output

    def get_sequence_num(self, preproc_data):
        return len(preproc_data[0])

    def format_output(self, preproc_data):
        #reward, actions, state
        output = {'reward': self.get_formatted_data(preproc_data, 'reward'), 
                'actions': self.get_formatted_data(preproc_data, 'action'),
                'state': self.get_formatted_data(preproc_data, 'state', format_as_tensor=False),
                'batch_size': len(preproc_data),
                'sequence_num': self.get_sequence_num(preproc_data)}
        return output

    def __getitem__(self, item):
        # print("in __getitem__")
        # print(item)
        # print(self.rb_data)
        if type(item) == slice:
            return self.format_output(self.rb_data[item])
        elif type(item) == list or type(item) == np.ndarray:
            output = []
            for i in item:
                output.append(self.rb_data[i])
            return self.format_output(output)
        else:
            logging.error("Unknown type in __getitem__ of Replay Buffer. Type: %s" % type(item))


class RLLearner():
    def __init__(self, rlc, args):
        self.args = args
        self.rlc = rlc

        self.params = list(rlc.parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_rlc = copy.deepcopy(rlc)

        # self.log_stats_t = -self.args.learner_log_interval - 1
    
    def train(self, episode_batch):
        # Get the relevant quantities
        rewards = episode_batch["reward"][:, :-1]
        actions = episode_batch["actions"][:, :-1]
        # terminated = batch["terminated"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = episode_batch["avail_actions"]

        # Calculate estimated Q-Values
        agent_outs_q = []
        agent_outs_i = []
        self.rlc.init_hidden(episode_batch['batch_size'])
        for t in range(episode_batch['sequence_num']):
            logging.debug("train forward passing for batch with time t=%s" % t)
            agent_out_q, agent_out_i  = self.rlc.forward(episode_batch, t=t)

            # print("agent_out_q")
            # print(agent_out_q.size())
            # print("agent_out_i")
            # print(agent_out_i.size())
            # print("agent_out size")
            # print(agent_outs.size())
            agent_outs_q.append(agent_out_q)
            agent_outs_i.append(agent_out_i)
        # agent_outs_q = torch.stack(agent_outs_q, dim=1)  # Concat over time
        # agent_outs_i = torch.stack(agent_outs_i, dim=1)

        # print("Actions")
        # print(actions)
        # print(actions.size())
        # print(len(agent_outs_i))
        # print(agent_outs_i[0].size())
        

        action_idxs = []
        best_qs = []
        for b_i in range(actions.size()[0]):
            # print("b_i: %s" % b_i)
            action_idxs_t = []
            best_qs_t = []
            for t_i in range(actions.size()[1]):
                # print("t idx: %s" % t_i)
                # print("action: %s" % actions[b_i][t_i][0])
                # idx_temp = list(agent_outs_i[t_i][b_i].data.numpy()).index(actions[b_i][t_i][0])
                # print("action idx: %s" % idx_temp)
                # print("action:")
                # print(agent_outs_i[t_i][b_i][idx_temp])
                idx = list(agent_outs_i[t_i][b_i].data.numpy()).index(actions[b_i][t_i][0])
                action_idxs_t.append(idx)
                best_qs_t.append(agent_outs_q[t_i][b_i][idx])
            action_idxs.append(action_idxs_t)
            best_qs.append(torch.stack(best_qs_t))
        # action_idxs = (actions == agent_outs_i).max(1)[1]
        action_idxs = torch.tensor(action_idxs)
        chosen_action_qvals = torch.stack(best_qs)

        # print("action_idxs")
        # print(action_idxs)
        # print(action_idxs.size())

        # print(actions)
        # print(actions.size())

        # print(agent_outs_i)
        # print(agent_outs_q)


        logging.debug("chosen action q values: %s" % chosen_action_qvals)


        # Calculate the Q-Values necessary for the target
        target_max_qvals = []
        # target_agent_outs_i = []
        self.target_rlc.init_hidden(episode_batch['batch_size'])
        for t in range(episode_batch['sequence_num']):
            target_agent_out_q, _  = self.target_rlc.forward(episode_batch, t=t)
            batch_best_qs = target_agent_out_q.max(dim=1)[0]
            target_max_qvals.append(batch_best_qs)
            # target_agent_outs_i.append(target_agent_out_i)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_max_qvals = torch.stack(target_max_qvals[1:], dim=1)  # Concat across time
        # target_agent_outs_i = torch.stack(target_agent_outs_[1:], dim=1)
        # Mask out unavailable actions
        # target_agent_outs[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        # if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # agent_outs[avail_actions == 0] = -9999999
            #TODO double not verified
            # cur_max_actions = agent_outs_q[:, 1:].max(dim=2, keepdim=True)[1]
            # target_max_qvals = torch.gather(target_agent_outs_q, 2, cur_max_actions).squeeze(3)
        # else:

        logging.debug("target max qvals: %s"  %target_max_qvals)
        
        rewards = rewards.squeeze(2)
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask

        # Normal L2 loss, 
        loss = (td_error ** 2).sum() 
        logging.info("RL Training loss %s" % loss)


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        #TODO remove this
        return
        #end todo

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            print("loss", loss.item(), t_env)
            print("grad_norm", grad_norm, t_env)
            
            print("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            print("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            print("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_q_net.load_state(self.q_net)
        print("Updated target network")

    def cuda(self):
        self.rlc.cuda()
        self.target_q_net.cuda()

    def save_models(self, path):
        self.rlc.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.rlc.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_q_net.load_models(path)
    
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class RLController():
    def __init__(self, args, q_net, train_dataset):
        self.rnn_hidden_state = None
        self.q_net = q_net
        self.train_dataset = train_dataset
        self.args = args
        

    @property
    def logger(self):
        return logging.getLogger(RLController.__name__)

    def init_hidden(self, batch_size):
        self.rnn_hidden_state = self.q_net.init_hidden().expand(batch_size, -1)

    def forward(self, episode_batch, t, test_mode=False):
        output_q = []
        output_i = []
        for episode_idx in range(len(episode_batch['state'])):

            unlabeled_indices = episode_batch['state'][episode_idx][t]

            unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices.data.numpy())

            unlabeled_dataloader_rl = data.DataLoader(self.train_dataset, 
                    sampler=unlabeled_sampler, batch_size=1, drop_last=False)

            all_q = []
            all_indices = []
            for images, _, indices in unlabeled_dataloader_rl:
                if self.args.cuda:
                    images = images.cuda()

                
                #TODO add other features to the network
                q, _ = self.q_net(images, self.rnn_hidden_state[episode_idx])

                all_q.extend(q)
                all_indices.extend(indices)

            all_q = torch.stack(all_q)
            all_q = all_q.view(-1)
            self.logger.debug("q values of size %s for episode %s: %s" % (all_q.size(), episode_idx, all_q))
            self.logger.debug("indices for q values: %s" % all_indices)
            output_q.append(all_q)
            output_i.append(torch.tensor(all_indices))
            
        output_q = torch.stack(output_q)
        output_i = torch.stack(output_i)
        
        return output_q, output_i

    def select_actions(self, episode_batch, t, test_mode=False):
        batch_qs, batch_is = self.forward(episode_batch, t, test_mode)

        sampled_indices_batch = []
        best_q_batch = []

        for i in range(batch_qs.size()[0]):
            all_q = batch_qs[i]
            all_indices = batch_is[i]

            if test_mode:
                epsilon = 0.0
            else:
                epsilon = self.args.epsilon

            random_numbers = torch.rand(1)
            pick_random = (random_numbers < epsilon).long()

            # Action selection
            if pick_random == 1:
                random_action_i_idx = Categorical(batch_is.float()).sample().long()
                sampled_indices = all_indices[random_action_i_idx].data.numpy()
                self.logger.debug("sampled random indices (ori dataset order): %s" % sampled_indices)
                print(type(sampled_indices))
            else:
                sampler_on_q = custom_sampler.TopSampler(self.args.budget)
                sampled_indices = sampler_on_q.sample(all_q, all_indices)
                self.logger.debug("sampled indices (original dataset order): %s" % sampled_indices)
                print(type(sampled_indices))
            best_q = None
            
            temp_sampler = data.sampler.SubsetRandomSampler(list(sampled_indices))
            best_q_image_dataloader = data.DataLoader(self.train_dataset, sampler=temp_sampler, 
                    batch_size=self.args.batch_size, drop_last=False)
            for images, _, indices in best_q_image_dataloader:
                # self.logger.debug("image: %s" % str(images))
                if self.args.cuda:
                    images = images.cuda()
                self.logger.debug("hidden state before update: %s" % self.rnn_hidden_state)
                best_q, self.rnn_hidden_state[i] = self.q_net(images, self.rnn_hidden_state[[i]])
                self.logger.debug("hidden state after update: %s" % self.rnn_hidden_state)
                self.logger.debug("best q: %s" % best_q)
            
            sampled_indices_batch.append(sampled_indices)
            best_q_batch.append(best_q)
        
        return sampled_indices_batch, best_q_batch

    def parameters(self):
        return self.q_net.parameters()


def take_step(all_indices, current_indices, train_dataset, querry_dataloader, solver):
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
    return acc, unlabeled_indices

def run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc, test_mode=False):
    sampler = data.sampler.SubsetRandomSampler(initial_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=False)

    #TODO change batch size to be configurable, maybe we can use it speed up the roll out as well
    rlc.init_hidden(1)

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

        logging.debug("RNN hidden input: %s" % str(rlc.rnn_hidden_state))
        
        acc, unlabeled_indices = take_step(all_indices, current_indices, train_dataset, querry_dataloader, solver)
        logging.info('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        logging.debug("unlabelled indices: %s" % unlabeled_indices)
        
        if len(accuracies) > 0:
            reward_for_last_step = acc - accuracies[-1]
            episode_info[-1]['reward'] = torch.tensor([reward_for_last_step/100.0])

        accuracies.append(acc)

        #reward is caculated in the next iteration of the loop
        step_execution_data = {'state': torch.tensor(unlabeled_indices), 'accurcy': torch.tensor([acc]), 't':t, 'reward': None}
        
        batch_execution_data = copy.copy(step_execution_data)
        batch_execution_data['state'] = batch_execution_data['state'].view(1, 1, -1)


        sampled_indices, best_q = rlc.select_actions(batch_execution_data, 0)

        step_execution_data['action'] = torch.tensor(sampled_indices[0])
        episode_info.append(step_execution_data)

        logging.debug("len of current_indices before adding sample indices: %s" % len(current_indices))
        current_indices = list(current_indices) + list(sampled_indices[0])
        logging.debug("current indices: %s" % current_indices)
        logging.debug("len of current_indices after adding sample indices: %s" % len(current_indices))
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=False)

    #Take the last step to get reward
    acc, unlabeled_indices = take_step(all_indices, current_indices, train_dataset, querry_dataloader, solver)
    if len(accuracies) > 0:
        reward_for_last_step = acc - accuracies[-1]
        episode_info[-1]['reward'] = torch.tensor([reward_for_last_step])
    accuracies.append(acc)

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
        args.num_images = 25
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
    rl_learner = RLLearner(rlc, args)


    num_runs = 100
    batch_size = 16
    for i in range(num_runs):
        accs, episode_info = run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc)

        logging.debug("epsode_info: %s"  % episode_info)

        replay_buffer.insert_episode(episode_info)

        if replay_buffer.can_sample(batch_size):
            episode_sample = replay_buffer.sample(batch_size)
            if args.cuda:
                episode_sample.cuda()

            logging.info("Start training...")
            rl_learner.train(episode_sample)

        # eval_interval = 20
        # if i % eval_interval == 0:
        #     run_episode(args, initial_indices, all_indices, train_dataset, solver, q_net, rlc)
        #TODO save RL model here
    



    # torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

