# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
from pathlib import Path
import os
import time

import torch
import src.distrib as distrib
from src.utils import bold, copy_state, LogProgress


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, data, model, criterion, optimizer, args, model_flops):
        self.tr_loader = data['tr']
        self.tt_loader = data['tt']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.flops_penalty = args.flops_penalty
        self.model_flops = model_flops        
        self.compressed_model_flops = model_flops

        if args.lr_sched == 'step':
            from torch.optim.lr_scheduler import StepLR
            sched = StepLR(self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma)
        elif args.lr_sched == 'multistep':
            from torch.optim.lr_scheduler import MultiStepLR
            sched = MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=args.multistep.gamma)
        elif args.lr_sched == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            sched = ReduceLROnPlateau(
                self.optimizer, factor=args.plateau.factor, patience=args.plateau.patience)
        elif args.lr_sched == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            sched = CosineAnnealingLR(
                self.optimizer, T_max=args.cosine.T_max, eta_min=args.cosine.min_lr)
        else:
            sched = None
        self.sched = sched

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm

        # Checkpoints
        self.continue_from = args.continue_from
        self.checkpoint = Path(
            args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s", self.checkpoint.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []

        # logging
        self.num_prints = args.num_prints

        if args.mixed:
            self.scaler = torch.cuda.amp.GradScaler()

        # for seperation tests
        self.args = args
        self._reset()

    def _serialize(self, path):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        if self.args.mixed:
            package['scaler'] = self.scaler.state_dict()
        if self.sched is not None:
            package['sched'] = self.sched.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint) + ".tmp"

        torch.save(package, tmp_path)
        os.rename(tmp_path, path)

    def _reset(self):
        load_from = None
        # Reset
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            strict = load_from == self.checkpoint
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=strict)
            else:
                self.model.load_state_dict(package['state'], strict=strict)
            if load_from == self.checkpoint:
                self.optimizer.load_state_dict(package['optimizer'])
                if self.args.mixed:
                    self.scaler.load_state_dict(package['scaler'])
                if self.sched is not None:
                    self.sched.load_state_dict(package['sched'])
                self.history = package['history']
                self.best_state = package['best_state']

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {info}")
            if self.sched is not None:
                self.sched.step()

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            if epoch == 0:
                print("Test pretrained original model")
                self.model.eval()
                self.test(ori_model=True)    
            
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            
            train_loss, train_acc = self._run_one_epoch(epoch)
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                              f'Time {time.time() - start:.2f}s | train loss {train_loss:.5f} | '
                              f'train accuracy {train_acc:.2f}'))
            
            # # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout & Diffq

            with torch.no_grad():
                valid_loss, valid_acc = self._run_one_epoch(epoch, cross_valid=True)
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | valid Loss {valid_loss:.5f} | '
                             f'valid accuracy {valid_acc:.2f}'))

            # learning rate scheduling
            if self.sched:
                if self.args.lr_sched == 'plateau':
                    self.sched.step(valid_loss)
                else:
                    self.sched.step()
                new_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info(f'Learning rate adjusted: {new_lr:.5f}')

            best_loss = float('inf')
            best_size = 0
            best_acc = 0
            for metrics in self.history:
                if metrics['valid'] < best_loss:
                    best_size = metrics['model_flops']
                    best_acc = metrics['valid_acc']
                    best_loss = metrics['valid']
            metrics = {'train': train_loss, 'train_acc': train_acc,
                       'valid': valid_loss, 'valid_acc': valid_acc,
                       'best': best_loss, 'best_size': best_size, 'best_acc': best_acc,
                       'compressed_model_flops': self.compressed_model_flops,
                       'model_flops': self.model_flops}

            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize(self.checkpoint)
                    logger.debug("Checkpoint saved to %s", self.checkpoint.resolve())


    def test(self, ori_model=False):
        # Optimizing the model
        self.model.eval()
        
        start = time.time()
        with torch.no_grad():
            valid_loss, valid_acc = self._run_one_epoch(0, cross_valid=True, ori_model=ori_model)
        print(f'Valid Summary | End of Epoch {1} | '
                            f'Time {time.time() - start:.2f}s | valid Loss {valid_loss:.5f} | '
                            f'valid accuracy {valid_acc:.2f}')
        logger.info(bold(f'Valid Summary | End of Epoch {1} | '
                            f'Time {time.time() - start:.2f}s | valid Loss {valid_loss:.5f} | '
                            f'valid accuracy {valid_acc:.2f}'))


        best_loss = float('inf')
        best_size = 0
        best_acc = 0
        for metrics in self.history:
            if metrics['valid'] < best_loss:
                best_size = metrics['model_flops']
                best_acc = metrics['valid_acc']
                best_loss = metrics['valid']
        metrics = {'valid': valid_loss, 'valid_acc': valid_acc,
                    'best': best_loss, 'best_size': best_size, 'best_acc': best_acc,
                    }

            # Save the best model
        if valid_loss == best_loss:
            logger.info(bold('New best valid loss %.4f'), valid_loss)
            self.best_state = copy_state(self.model.state_dict())

    def _run_one_epoch(self, epoch, cross_valid=False, ori_model=False):
        total_loss = 0
        avg_flops = 0
        total = 0
        correct = 0
        data_loader = self.tr_loader if not cross_valid else self.tt_loader
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, (inputs, targets) in enumerate(logprog):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if not cross_valid:
                with torch.cuda.amp.autocast(bool(self.args.mixed)):
                    yhat, gumbel_idx = self.dmodel(inputs)
                    loss = self.criterion(yhat, targets)
                    model_flops = self.dmodel.module.compute_flops(gumbel_idx)
                    if self.flops_penalty > 0:
                        loss = loss + self.flops_penalty * model_flops.mean()
            else:
                # compute output
                if ori_model:
                    if hasattr(self.dmodel, 'module'):
                        if hasattr(self.dmodel.module, 'forward_original'):
                            
                            yhat = self.dmodel.module.forward_original(inputs)
                    elif hasattr(self.dmodel, 'forward_original'):
                        yhat = self.dmodel.forward_original(inputs)                    
                    else:
                        yhat = self.dmodel(inputs)
                else:
                    yhat, gumbel_idx = self.dmodel(inputs)
                loss = self.criterion(yhat, targets)

            if not cross_valid:
                # optimize model in training mode
                self.optimizer.zero_grad()
                
                if self.args.mixed:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                if self.args.mixed:
                    self.scaler.step(self.optimizer)
                else:
                    self.optimizer.step()
                
                if self.args.mixed:
                    self.scaler.update()

            total_loss += loss.item()
            _, predicted = yhat.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_acc = 100. * (correct / total)
            if cross_valid:
                print(f"total_loss : {total_loss} total_acc : {total_acc}")
            if not cross_valid:
                logprog.update(
                    loss=format(total_loss / (i + 1), ".5f"),
                    accuracy=format(total_acc, ".5f"), Average_FLOPS=format(model_flops.mean().item(), ".3f"))
            else:
                logprog.update(loss=format(total_loss / (i + 1), ".5f"),
                               accuracy=format(total_acc, ".5f"))
            # Just in case, clear some memory
            del loss
        return (distrib.average([total_loss / (i + 1)], i + 1)[0],
                distrib.average([total_acc], total)[0])
