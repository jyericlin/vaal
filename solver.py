import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler

from tqdm import tqdm


class ReplayBuffer():
    def __init__(self):
        self.episode_data = {}

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    #TODO implement
    def insert_episode(self, episode):
        pass

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    #TODO implement
    def __getitem__(self, item):
        print("in __getitem__")
        print(item)


class RLLearner():
    def __init__(self):
        pass
    
    def train(self, episode_batch):
        pass



class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.is_multi_class = args.is_multi_class

        self.sampling_method = args.sampling_method
        if self.sampling_method == "random":
            self.sampler = sampler.RandomSampler(self.args.budget)
        elif self.sampling_method == "adversary":
            self.sampler = sampler.AdversarySampler(self.args.budget)
        elif self.sampling_method == "uncertainty":
            self.sampler = sampler.UncertaintySampler(self.args.budget)
        elif self.sampling_method == "adversary_1c":
            self.sample = sampler.AdversarySamplerSingleClass(self.args.budget)
        else:
            raise Exception("No valid sampling method provideds")



    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def controller(self,):
        pass

    def run_episode(self, controller, querry_dataloader, unlabeled_dataloader, task_model, test_mode=False):
        pass
    

    def run(self, num_runs, replay_buffer, learner, controller, batch_size):
        for i in num_runs:
            episode = self.run_episode(controller)
            replay_buffer.insert_episode(episode)

            if replay_buffer.can_sample(batch_size):
                episode_sample = replay_buffer.sample(batch_size)
                #check if CUDA exist here
                episode_sample.cuda()

                learner.train(episode_sample)

            eval_interval = 30
            if i%eval_interval == 0:
                self.run_episode(controller, test_mode=True)
            #TODO save RL model here


            

    def train_RL_without_adv_vae(self, querry_dataloader, task_model, unlabeled_dataloader):

        final_acc = self.train_without_adv_vae(querry_dataloader, task_model, None, None, unlabeled_dataloader)



    def train_without_adv_vae(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):

        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
    
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9 

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 100 == 0:
                print('Current task model loss: {:.4f}'.format(task_loss.item()))


        final_accuracy = self.test(task_model)
        return final_accuracy, vae, discriminator
    

    def train(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):

        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] * 0.9
    
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9 

                for param in optim_discriminator.param_groups:
                    param['lr'] = param['lr'] * 0.9 

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            # VAE step
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                if self.is_multi_class:
                    lab_real_preds = torch.zeros(labeled_imgs.size(0)).long()
                    unlab_real_preds = torch.zeros(unlabeled_imgs.size(0)).long()
                else:
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                    
                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.ce_loss(labeled_preds, lab_real_preds) + \
                        self.ce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                # labeled_preds = labeled_out.max(1)[1]
                unlabeled_preds = discriminator(unlab_mu)
                # unlabeled_preds = unlabeled_out.max(1)[1]
                
                if self.is_multi_class:
                    lab_real_preds = labels
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).long()
                else:
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))


                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = self.ce_loss(labeled_preds, lab_real_preds) + \
                        self.ce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

                

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        final_accuracy = self.test(task_model)
        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, task_learner):
        if self.sampling_method == "random":
            querry_indices = self.sampler.sample(unlabeled_dataloader)
        elif self.sampling_method == "uncertainty":
            querry_indices = self.sampler.sample(task_learner, 
                                                unlabeled_dataloader, 
                                                self.args.cuda)
        elif self.sampling_method == "adversary" or self.sampling_method == "adversary_1c":
            querry_indices = self.sampler.sample(vae, 
                                                discriminator, 
                                                unlabeled_dataloader, 
                                                self.args.cuda)

        return querry_indices
                

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
