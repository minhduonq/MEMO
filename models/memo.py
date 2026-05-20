
import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.relative import relative_geometry_loss_per_class
from utils.prototype import prototype_regularization_loss

num_workers=8

class MEMO(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args['convnet_type'], False)
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

        self._buffer_feat_old = None  
        self._buffer_labels = None     
        self._buffer_imgs = None       
        self._buffer_trsf = None       
        self._buffer_num_extractors = 0  

        self._prototypes = None        

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

        self._update_geometry_buffer()
        self._update_prototypes()


        self._save_feature_snapshot()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train', 
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', 
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._buffer_trsf = transforms.Compose(
            [*data_manager._test_trsf, *data_manager._common_trsf]
        )
        self._buffer_use_path = data_manager.use_path
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

    def _update_geometry_buffer(self):
        if len(self._data_memory) == 0:
            logging.info("No exemplar buffer yet, skipping geometry buffer update.")
            return

        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        self._buffer_num_extractors = len(net.AdaptiveExtractors)

        logging.info("Updating geometry buffer ({} exemplars, {} extractors)...".format(
            len(self._data_memory), self._buffer_num_extractors))

        self._buffer_imgs = copy.deepcopy(self._data_memory)     
        self._buffer_labels = torch.tensor(
            self._targets_memory.astype(int), dtype=torch.long, device=self._device
        )

        full_feat = self._compute_full_features(self._buffer_imgs)
        self._buffer_feat_old = full_feat.detach()  

        logging.info("Geometry buffer updated. feat shape: {}".format(
            self._buffer_feat_old.shape
        ))

    def _update_prototypes(self):
        if self._buffer_feat_old is None or self._buffer_labels is None:
            logging.info("No geometry buffer yet, skipping prototype update.")
            return

        feat = self._buffer_feat_old   
        labels = self._buffer_labels   
        max_class = int(labels.max().item()) + 1
        D = feat.shape[1]

        prototypes = torch.zeros(max_class, D, device=feat.device)
        for c in range(max_class):
            mask = (labels == c)
            if mask.sum() > 0:
                prototypes[c] = feat[mask].mean(dim=0)

        self._prototypes = prototypes.detach()
        logging.info("Prototypes updated. shape: {} ({} classes)".format(
            self._prototypes.shape, max_class
        ))



    def _save_feature_snapshot(self):
        save_dir = os.path.join(
            self.args.get('logfilename', 'logs'), 'features', f'task_{self._cur_task}'
        )
        os.makedirs(save_dir, exist_ok=True)

        if self._buffer_feat_old is not None:
            torch.save(self._buffer_feat_old.cpu(), os.path.join(save_dir, 'features.pt'))
            torch.save(self._buffer_labels.cpu(), os.path.join(save_dir, 'labels.pt'))
            logging.info(f"Saved feature snapshot to {save_dir}")

        if self._prototypes is not None:
            torch.save(self._prototypes.cpu(), os.path.join(save_dir, 'prototypes.pt'))
            logging.info(f"Saved prototypes to {save_dir}")

    def _compute_full_features(self, image_data, num_extractors=None):
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        net.eval()
        if num_extractors is None:
            num_extractors = len(net.AdaptiveExtractors)

        all_feats = []
        trsf = self._buffer_trsf
        use_path = getattr(self, '_buffer_use_path', False)

        bs = 64
        with torch.no_grad():
            for start in range(0, len(image_data), bs):
                batch_data = image_data[start:start+bs]
                imgs = []
                for item in batch_data:
                    if use_path:
                        from utils.data_manager import pil_loader
                        img = trsf(pil_loader(item))
                    else:
                        img = trsf(Image.fromarray(item))
                    imgs.append(img)
                imgs_tensor = torch.stack(imgs).to(self._device)

                base_fmap = net.TaskAgnosticExtractor(imgs_tensor)
                features = [net.AdaptiveExtractors[i](base_fmap)
                            for i in range(num_extractors)]
                feat_vec = torch.cat(features, dim=1) 
                feat_vec = F.normalize(feat_vec, p=2, dim=1)
                all_feats.append(feat_vec)

        return torch.cat(all_feats, dim=0)

    def _sample_buffer_subset(self, n_samples):
        total = len(self._buffer_imgs)
        if total <= n_samples:
            return self._buffer_imgs, self._buffer_labels, self._buffer_feat_old

        idx = np.random.choice(total, n_samples, replace=False)
        sub_imgs = self._buffer_imgs[idx]
        sub_labels = self._buffer_labels[idx]
        sub_feat_old = self._buffer_feat_old[idx]
        return sub_imgs, sub_labels, sub_feat_old

    def _forward_full_features_with_grad(self, image_data, num_extractors):
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        trsf = self._buffer_trsf
        use_path = getattr(self, '_buffer_use_path', False)

        imgs = []
        for item in image_data:
            if use_path:
                from utils.data_manager import pil_loader
                img = trsf(pil_loader(item))
            else:
                img = trsf(Image.fromarray(item))
            imgs.append(img)
        imgs_tensor = torch.stack(imgs).to(self._device)

        base_fmap = net.TaskAgnosticExtractor(imgs_tensor)
        features = [net.AdaptiveExtractors[i](base_fmap)
                    for i in range(num_extractors)]
        feat_vec = torch.cat(features, dim=1)  
        feat_vec = F.normalize(feat_vec, p=2, dim=1)
        return feat_vec

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, 
                    milestones=self.args['init_milestones'], 
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                ) 
            else:
                raise NotImplementedError
            
            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                lr=self.args['lrate'], 
                momentum=0.9, 
                weight_decay=self.args['weight_decay']
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'], 
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            # prog_bar.set_description(info)
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))

        # Geometry loss config
        alpha_relative_max = self.args.get('alpha_relative', 0.6)
        alpha_prototype = self.args.get('alpha_prototype', 0.3)

        geo_loss_freq = self.args.get('geo_loss_freq', 1)
        geo_buffer_sample = self.args.get('geo_buffer_sample', 128)
        geo_max_classes = self.args.get('geo_max_classes', 10)
        geo_anchors = self.args.get('geo_anchors', 2)
        # Warmup schedule: gradually phase in geometry losses
        warmup_start = self.args.get('geo_warmup_start', 30)
        warmup_end = self.args.get('geo_warmup_end', 80)
        has_geo_buffer = self._buffer_feat_old is not None
        has_prototypes = self._prototypes is not None


        for _, epoch in enumerate(prog_bar):
            self.set_network()

            # Compute warmup factor for geometry losses
            if epoch < warmup_start:
                geo_factor = 0.0
            elif epoch < warmup_end:
                geo_factor = (epoch - warmup_start) / (warmup_end - warmup_start)
            else:
                geo_factor = 1.0
            alpha_relative = alpha_relative_max * geo_factor
            

            losses = 0.
            losses_clf=0.
            losses_aux=0.
            losses_rel_sum = 0.
            losses_proto_sum = 0.

            
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs= self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                loss_clf=F.cross_entropy(logits,targets)
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                loss_aux=F.cross_entropy(aux_logits,aux_targets)
                loss=loss_clf+self.args['alpha_aux']*loss_aux
                loss_rel_val = 0.
                loss_proto_val = 0.

                
                if has_geo_buffer and (i % geo_loss_freq == 0):
                    sub_imgs, sub_labels, sub_feat_old = self._sample_buffer_subset(geo_buffer_sample)

                    z_new = self._forward_full_features_with_grad(
                        sub_imgs, self._buffer_num_extractors) 
                    z_old = sub_feat_old.to(self._device)        

                    L_rel = relative_geometry_loss_per_class(
                        z_old=z_old,
                        z_new=z_new,
                        y_old=sub_labels,
                        max_classes_per_step=geo_max_classes,
                        anchors_per_class=geo_anchors,
                    )

                    loss = loss + alpha_relative * L_rel 
                    loss_rel_val = L_rel.item()

                    if has_prototypes:
                        L_proto = prototype_regularization_loss(
                            z_new=z_new,
                            y=sub_labels,
                            prototypes=self._prototypes,
                        )
                        loss = loss + alpha_prototype * L_proto
                        loss_proto_val = L_proto.item()



                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=5.0)
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()
                losses_rel_sum += loss_rel_val
                losses_proto_sum += loss_proto_val


                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            n_batches = len(train_loader)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = ('Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, '
                        'Loss_rel {:.3f}, Loss_proto {:.5f}, geo_w {:.2f}, Train_accy {:.2f}, Test_accy {:.2f}').format(
                    self._cur_task, epoch+1, self.args["epochs"],
                    losses/n_batches, losses_clf/n_batches, losses_aux/n_batches,
                    losses_rel_sum/n_batches, losses_proto_sum/n_batches,
                    geo_factor, train_acc, test_acc)
            else:
                info = ('Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, '
                        'Loss_rel {:.3f}, Loss_proto {:.5f}, geo_w {:.2f}, Train_accy {:.2f}').format(
                    self._cur_task, epoch+1, self.args["epochs"],
                    losses/n_batches, losses_clf/n_batches, losses_aux/n_batches,
                    losses_rel_sum/n_batches, losses_proto_sum/n_batches,
                    geo_factor,  train_acc)
            prog_bar.set_description(info)
        logging.info(info)