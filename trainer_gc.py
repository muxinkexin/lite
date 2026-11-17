from __future__ import absolute_import, division, print_function
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from networks.pose_encoder import create_pose_encoder


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []


        self.device = select_device(self.opt.no_cuda)

        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.models["encoder"] = networks.EfficientEncoder(valid_models=self.opt.model,
                                                           height=self.opt.height, width=self.opt.width)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,iters = self.opt.iters)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        

        if self.use_pose_net:
            self.models_pose["pose_encoder"] = create_pose_encoder(enc_name = 'efficientnet_b1',num_inputs=2, use_coords=True)
            self.models_pose["pose_encoder"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

            self.models_pose["pose"] = networks.PoseDecoder(
                self.models_pose["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.opt.lr[2]),
                            T_mul=1,
                            eta_min=self.opt.lr[1],
                            last_epoch=-1,
                            max_lr=self.opt.lr[0],
                            warmup_steps=0,
                            gamma=0.9
                        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=(self.device.type == "cuda"), drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=(self.device.type == "cuda"), drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        scale = 0
        h = self.opt.height // (2 ** scale)
        w = self.opt.width // (2 ** scale)

        self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
        self.backproject_depth[scale].to(self.device)

        self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
        self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
    
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
    
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
    
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):

        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            self.model_pose_optimizer.step()
            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #self.val()
            if self.epoch>20 and self.epoch<40 and (self.step % 1000) ==0 :
                self.save_model()
            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features, iters = self.opt.iters)

        aa = self.models["encoder"](inputs["color_aug", -1, 0])
        bb = self.models["depth"](aa, iters = self.opt.iters)
        outputs_1 = {}
        outputs_1[("disp_up", -1, 0)] = bb[("disp_up", 0)]
        outputs_1[("disp_up", -1, 1)] = bb[("disp_up", 1)]
        #outputs_1[("disp_up", -1, 2)] = bb[("disp_up", 2)]
        #outputs_1[("disp_up", -1, 3)] = bb[("disp_up", 3)]
        #outputs_1[("disp_up", -1, 4)] = bb[("disp_up", 4)]
        #outputs_1[("disp_up", -1, 5)] = bb[("disp_up", 5)]

        aa = self.models["encoder"](inputs["color_aug", 1, 0])
        bb = self.models["depth"](aa, iters = self.opt.iters)
        outputs1 = {}
        outputs1[("disp_up", 1, 0)] = bb[("disp_up", 0)]
        outputs1[("disp_up", 1, 1)] = bb[("disp_up", 1)]
        #outputs1[("disp_up", 1, 2)] = bb[("disp_up", 2)]
        #outputs1[("disp_up", 1, 3)] = bb[("disp_up", 3)]
        #outputs1[("disp_up", 1, 4)] = bb[("disp_up", 4)]
        #outputs1[("disp_up", 1, 5)] = bb[("disp_up", 5)]
        outputs1.update(outputs_1)
        outputs.update(outputs1)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))

        self.generate_images_pred(inputs, outputs, iters = self.opt.iters)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models_pose["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, iters=3):

        for scale in range(iters):

            disp = outputs[("disp_up", scale)]

            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords, computed_depth = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("computed_depth", frame_id, scale)] = computed_depth
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                _, outputs[("projected_depth", frame_id, scale)] = disp_to_depth(F.grid_sample(
                    outputs[("disp_up", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border"), self.opt.min_depth, self.opt.max_depth)

                
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]
    
    def smooth_l1_loss_ours(self, input,target,beta=0.15):
        n = torch.abs(input-target)
        cond = n<beta
        loss = torch.where(cond, 0.5*n**2, n)
        return loss
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        SmoothL1_loss = self.smooth_l1_loss_ours(pred,target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * SmoothL1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        
        for scale in range(self.opt.iters):
            loss = 0
            reprojection_losses = []
            i_weight = 0.9**(self.opt.iters - scale - 1)

            diff_depth_1 = ((outputs[("computed_depth", -1, scale)] - outputs[("projected_depth", -1, scale)]).abs() / (outputs[("computed_depth", -1, scale)] + outputs[("projected_depth", -1, scale)]).abs())
            
            diff_depth1 = ((outputs[("computed_depth", 1, scale)] - outputs[("projected_depth", 1, scale)]).abs() / (outputs[("computed_depth", 1, scale)] + outputs[("projected_depth", 1, scale)]).abs())
            
            source_scale = 0

            disp = outputs[("disp_up", scale)]
            color = inputs[("color", 0, source_scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
            identity_reprojection_loss1, idxs = torch.min(identity_reprojection_loss, dim=1)
            identity_reprojection_loss = torch.stack((identity_reprojection_loss1, identity_reprojection_loss1), dim=1)
            
            sta = reprojection_loss<identity_reprojection_loss
            beta = self.opt.beta

            reprojection_loss[sta==0] =100

            to_optimise = torch.where((reprojection_loss[:,0]<=beta) + (reprojection_loss[:,1]<=beta),
                            torch.where(reprojection_loss[:,0] < reprojection_loss[:,1],
                                        reprojection_loss[:,0],reprojection_loss[:,1]),
                            torch.zeros_like(reprojection_loss[:,0]))
            
            loss += i_weight*to_optimise.sum()/(to_optimise != 0).sum()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss * i_weight

            diff_loss = torch.where((reprojection_loss[:,0]<=beta) + (reprojection_loss[:,1]<=beta),
                            torch.where(reprojection_loss[:,0] < reprojection_loss[:,1],
                                        diff_depth_1.squeeze(),diff_depth1.squeeze()),
                            torch.zeros_like(reprojection_loss[:,0]))

            diff_loss = diff_loss.sum()/(diff_loss != 0).sum() * i_weight *0.1
            loss += diff_loss
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i].detach().cpu().numpy()
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0
            writer.add_image(
                "color_{}_{}/{}".format(s, s, j),
                inputs[("color", s, s)][j].data, self.step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp_up", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        #pretrained_dict = torch.load(path, map_location="cpu", weights_only=True)#['model']
        pretrained_dict = torch.load(path, map_location="cpu", weights_only=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model = self.models[n] if n in self.models else self.models_pose[n]
            model_dict = model.state_dict()
            pretrained_dict = torch.load(path, map_location="cpu", weights_only=True)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=True)###strict = True

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device, weights_only=True)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path, map_location=self.device, weights_only=True)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")