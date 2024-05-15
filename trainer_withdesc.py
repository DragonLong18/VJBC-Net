import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component,get_metrics_new,get_metrics_new_two
import ttach as tta
from models.utils import get_targetbb,get_gaussian_kernel
from torch.nn import functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import copy
class Trainer:
    def __init__(self, model ,CFG=None, loss=None,vessel_loss=None, train_loader=None, val_loader=None,soft_limit = True):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.vessel_loss = vessel_loss
        self.soft_limit = soft_limit
        if self.soft_limit:
            self.l1_loss = nn.L1Loss() 
        self.model = nn.DataParallel(model.cuda())  
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir = os.path.join(
            CFG.save_dir, self.CFG['model']['type'], start_time)
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True
        self.kernel = get_gaussian_kernel(7,2.1)


    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        wrt_mode = 'train'
        self._reset_metrics_two()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()
        for img, gt_vessel,gt_bifurcation, gt_cross in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt_bifurcation5 = F.conv2d(gt_bifurcation, self.kernel,
                             stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
            gt_cross5 = F.conv2d(gt_cross, self.kernel,
                             stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
            gt_bifurcation5 = gt_bifurcation5.cuda(non_blocking=True)
            gt_cross5 = gt_cross5.cuda(non_blocking=True)
            gt_vessel = gt_vessel.cuda(non_blocking=True)
            # print("type",type(gt5))
            self.optimizer.zero_grad()
            if self.CFG.amp is True:
                with torch.cuda.amp.autocast(enabled=True):
                    pre_vessel,pre_bifurcation, pre_cross,vessel_desc,bif_desc,cross_desc = self.model(img)
                    bifurcation_loss = self.loss(pre_bifurcation,gt_bifurcation5)
                    cross_loss = self.loss(pre_cross,gt_cross5)
                    vessel_losss = self.vessel_loss(pre_vessel,gt_vessel)
                    loss = bifurcation_loss + cross_loss + vessel_losss
                    if self.soft_limit:
                        loss = loss + 0.2*self.l1_loss(vessel_desc,bif_desc) + 0.2*self.l1_loss(vessel_desc,cross_desc)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pre_vessel,pre_bifurcation, pre_cross,vessel_desc,bif_desc,cross_desc = self.model(img)
                bifurcation_loss = self.loss(pre_bifurcation,gt_bifurcation5)
                cross_loss = self.loss(pre_cross,gt_cross5)
                vessel_losss = self.vessel_loss(pre_vessel,gt_vessel)
                loss = bifurcation_loss + cross_loss + vessel_losss
                if self.soft_limit:
                    loss = loss + 0.2*self.l1_loss(vessel_desc,bif_desc) + 0.2*self.l1_loss(vessel_desc,cross_desc)
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            
            self._metrics_update_two(
                *get_metrics_new_two(pre_bifurcation, gt_bifurcation, pre_cross, gt_cross, threshold=self.CFG.threshold).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | Bifurcation_AUC {:.4f} Bifurcation_F1 {:.4f} Bifurcation_Acc {:.4f}  Bifurcation_Sen {:.4f} Bifurcation_Spe {:.4f} Bifurcation_Pre {:.4f} Bifurcation_IOU {:.4f} Cross_AUC {:.4f} Cross_F1 {:.4f} Cross_Acc {:.4f}  Cross_Sen {:.4f} Cross_Spe {:.4f} Cross_Pre {:.4f} Cross_IOU {:.4f}  |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave_two().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave_two().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics_two()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for img,gt_vessel, gt_bifurcation, gt_cross in tbar:
                img = img.cuda(non_blocking=True)
                gt_bifurcation5 = F.conv2d(gt_bifurcation, self.kernel,
                             stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
                gt_cross5 = F.conv2d(gt_cross, self.kernel,
                             stride=1, padding=(self.kernel.shape[-1] - 1) // 2)
                gt_bifurcation5 = gt_bifurcation5.cuda(non_blocking=True)
                gt_cross5 = gt_cross5.cuda(non_blocking=True)
                gt_vessel = gt_vessel.cuda(non_blocking=True)
                if self.CFG.amp is True:
                    with torch.cuda.amp.autocast(enabled=True):
                        pre_vessel,pre_bifurcation, pre_cross,vessel_desc,bif_desc,cross_desc = self.model(img)
                        vessel_loss = self.vessel_loss(pre_vessel,gt_vessel)
                        bifurcation_loss = self.loss(pre_bifurcation,gt_bifurcation5)
                        cross_loss = self.loss(pre_cross,gt_cross5)
                        loss = bifurcation_loss + cross_loss + vessel_loss
                        if self.soft_limit:
                            loss = loss + 0.2*self.l1_loss(vessel_desc,bif_desc) + 0.2*self.l1_loss(vessel_desc,cross_desc)

                else:
                    pre_vessel,pre_bifurcation, pre_cross,vessel_desc,bif_desc,cross_desc = self.model(img)
                    vessel_loss = self.vessel_loss(pre_vessel,gt_vessel)
                    bifurcation_loss = self.loss(pre_bifurcation,gt_bifurcation5)
                    cross_loss = self.loss(pre_cross,gt_cross5)
                    loss = bifurcation_loss + cross_loss + vessel_loss
                    if self.soft_limit:
                        loss = loss + 0.2*self.l1_loss(vessel_desc,bif_desc) + 0.2*self.l1_loss(vessel_desc,cross_desc)
                self.total_loss.update(loss.item())
                self._metrics_update_two(
                    *get_metrics_new_two(pre_bifurcation, gt_bifurcation, pre_cross, gt_cross ,threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} |  Bifurcation_AUC {:.4f} Bifurcation_F1 {:.4f} Bifurcation_Acc {:.4f}  Bifurcation_Sen {:.4f} Bifurcation_Spe {:.4f} Bifurcation_Pre {:.4f} Bifurcation_IOU {:.4f} Cross_AUC {:.4f} Cross_F1 {:.4f} Cross_Acc {:.4f}  Cross_Sen {:.4f} Cross_Spe {:.4f} Cross_Pre {:.4f} Cross_IOU {:.4f}|'.format(
                        epoch, self.total_loss.average, *self._metrics_ave_two().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave_two().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave_two()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,
                                f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
        
    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }
        


    def _reset_metrics_two(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.bifurcation_auc = AverageMeter()
        self.bifurcation_f1 = AverageMeter()
        self.bifurcation_acc = AverageMeter()
        self.bifurcation_sen = AverageMeter()
        self.bifurcation_spe = AverageMeter()
        self.bifurcation_pre = AverageMeter()
        self.bifurcation_iou = AverageMeter()
        self.bifurcation_CCC = AverageMeter()
        self.cross_auc = AverageMeter()
        self.cross_f1 = AverageMeter()
        self.cross_acc = AverageMeter()
        self.cross_sen = AverageMeter()
        self.cross_spe = AverageMeter()
        self.cross_pre = AverageMeter()
        self.cross_iou = AverageMeter()
        self.cross_CCC = AverageMeter()
        # self.total_pre = AverageMeter()
        # self.total_sen = AverageMeter()
        # self.total_f1 = AverageMeter()
    def _metrics_update_two(self, bifurcation_auc, bifurcation_f1, bifurcation_acc, bifurcation_sen, bifurcation_spe, bifurcation_pre, bifurcation_iou, \
                            cross_auc, cross_f1,cross_acc,cross_sen, cross_spe, cross_pre, cross_iou):
        self.bifurcation_auc.update(bifurcation_auc)
        self.bifurcation_f1.update(bifurcation_f1)
        self.bifurcation_acc.update(bifurcation_acc)
        self.bifurcation_sen.update(bifurcation_sen)
        self.bifurcation_spe.update(bifurcation_spe)
        self.bifurcation_pre.update(bifurcation_pre)
        self.bifurcation_iou.update(bifurcation_iou)
        self.cross_auc.update(cross_auc)
        self.cross_f1.update(cross_f1)
        self.cross_acc.update(cross_acc)
        self.cross_sen.update(cross_sen)
        self.cross_spe.update(cross_spe)
        self.cross_pre.update(cross_pre)
        self.cross_iou.update(cross_iou)
        # self.total_pre.update(total_pre)
        # self.total_sen.update(total_sen) 
        # self.total_f1.update(total_f1) 
              
    def _metrics_ave_two(self):

        return {
            "Bifurcation_AUC": self.bifurcation_auc.average,
            "Bifurcation_F1": self.bifurcation_f1.average,
            "Bifurcation_Acc": self.bifurcation_acc.average,
            "Bifurcation_Sen": self.bifurcation_sen.average,
            "Bifurcation_Spe": self.bifurcation_spe.average,
            "Bifurcation_pre": self.bifurcation_pre.average,
            "Bifurcation_IOU": self.bifurcation_pre.average,
            "Cross_AUC": self.cross_auc.average,
            "Cross_F1": self.cross_f1.average,
            "Cross_Acc": self.cross_acc.average,
            "Cross_Sen": self.cross_sen.average,
            "Cross_Spe": self.cross_spe.average,
            "Cross_pre": self.cross_pre.average,
            "Cross_IOU": self.cross_pre.average,
            # "Total_pre": self.total_pre,
            # "Total_sen": self.total_sen,
            # "Total_f1": self.total_f1
        }

