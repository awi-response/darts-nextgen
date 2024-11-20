import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

import darts_superresolution.model.networks as networks
from darts_superresolution.model.base_model import BaseModel
from darts_superresolution.model.scheduler import CosineAnnealingWithDecay

logger = logging.getLogger(__name__)


class DDPM(BaseModel):
    def __init__(self, opt, device):
        super(DDPM, self).__init__(opt, device)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_net(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt["model"]["beta_schedule"]["train"], schedule_phase="train")
        if self.opt["phase"] == "train":
            self.netG.train()
            # find the parameters to optimize
            if opt["model"]["finetune_norm"]:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find("transformer") >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(f"Params [{k:s}] initialized to 0 and will optimize.")
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt["train"]["optimizer"]["lr"])
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optG, milestones=[2000, 10000, 20000, 50000], gamma=0.5)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optG, 20000)
            # self.scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(self.optG, T_0=30000, T_mult=1, warmup_steps=0, decay=0.3)
            # self.scheduler_1 = torch.optim.lr_scheduler.ConstantLR(self.optG, factor=1.0, total_iters=500)
            # self.scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, 10000, eta_min=0.00002, last_epoch=5)
            # self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optG, schedulers=[self.scheduler_1, self.scheduler_2], milestones=[500])
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, 7000)
            self.scheduler = CosineAnnealingWithDecay(self.optG, 75000)
            # self.scheduler = CosineAnnealingWithCosineRestarts(self.optG, 1000, eta_min=0.000002)#, decay=0.8)
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_fake_before_feed.tif", np.asarray(data['SR']))
        # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_real_before_feed.tif", np.asarray(data['HR']))
        self.data = self.set_device(data)
        # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_fake_after_feed.tif", np.asarray(self.data['SR'].detach().cpu()))
        # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/1/image_real_after_feed.tif", np.asarray(self.data['HR'].detach().cpu()))

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data["HR"].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict["l_pix"] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                # print("In: ", self.data['SR'].shape)
                self.SR = self.netG.module.super_resolution(self.data["SR"], continous)
                # print("Out: ", self.SR.shape)
            else:
                # print("In: ", self.data['SR'].shape)
                self.SR = self.netG.super_resolution(self.data["SR"], continous)
                # print("Out: ", self.SR.shape)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase="train"):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict["SAM"] = self.SR.detach().float().cpu()
        else:
            out_dict["SR"] = self.SR.detach().float().cpu()
            out_dict["INF"] = self.data["SR"].detach().float().cpu()
            out_dict["HR"] = self.data["HR"].detach().float().cpu()
            if need_LR and "LR" in self.data:
                out_dict["LR"] = self.data["LR"].detach().float().cpu()
            else:
                out_dict["LR"] = out_dict["INF"]
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = f"{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}"
        else:
            net_struc_str = f"{self.netG.__class__.__name__}"

        logger.info(f"Network G structure: {net_struc_str}, with parameters: {n:,d}")
        logger.info(s)

    def save_network(self, epoch, iter_step, best_ckp=False):
        if best_ckp == False:
            gen_path = os.path.join(self.opt["path"]["checkpoint"], f"I{iter_step}_E{epoch}_gen.pth")
            opt_path = os.path.join(self.opt["path"]["checkpoint"], f"I{iter_step}_E{epoch}_opt.pth")
        else:
            gen_path = os.path.join(self.opt["path"]["checkpoint"], f"I{iter_step}_E{epoch}_best_gen.pth")
            opt_path = os.path.join(self.opt["path"]["checkpoint"], f"I{iter_step}_E{epoch}_best_opt.pth")
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {"epoch": epoch, "iter": iter_step, "scheduler": None, "optimizer": None}
        opt_state["optimizer"] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(f"Saved model in [{gen_path:s}] ...")

    def load_network(self):
        load_path = self.opt["path"]["resume_state"]
        if load_path is not None:
            logger.info(f"Loading pretrained model for G [{load_path:s}] ...")
            gen_path = f"{load_path}_gen.pth"
            opt_path = f"{load_path}_opt.pth"
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            # network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
            network.load_state_dict(torch.load(gen_path), strict=False)
            if self.opt["phase"] == "train":
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt["optimizer"])
                self.begin_step = opt["iter"]
                self.begin_epoch = opt["epoch"]
