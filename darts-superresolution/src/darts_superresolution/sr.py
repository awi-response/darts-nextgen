import argparse
import logging
import os

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
import numpy as np
import torch
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config/sr_sr3_16_128.json", help="JSON file for configuration"
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["train", "val"],
        help="Run either train(training) or val(generation)",
        default="train",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_wandb_ckpt", action="store_true")
    parser.add_argument("-log_eval", action="store_true")

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
    Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=os.path.join(opt["path"]["tb_logger"], opt["name"]))

    # Initialize WandbLogger
    if opt["enable_wandb"]:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric("validation/val_step")
        wandb.define_metric("epoch")
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train" and args.phase != "val":
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == "val":
            dataset_opt["data_len"] = 8
            dataset_opt["batch_size"] = 1
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info("Initial Dataset Finished")

    # model
    diffusion = Model.create_model(opt)
    logger.info("Initial Model Finished")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt["train"]["n_iter"]

    if opt["path"]["resume_state"]:
        logger.info(f"Resuming training from epoch: {current_epoch}, iter: {current_step}.")

    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"][opt["phase"]], schedule_phase=opt["phase"])
    torch.manual_seed(42)
    best_psnr = 0
    best_checkpoint = ""
    best_opt_checkpoint = ""
    if opt["phase"] == "train":
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # diffusion.scheduler.step(current_step) # For pytorch cosine scheduler
                diffusion.scheduler.step()  # For custom scheduler with warmup and decay
                tb_logger.add_scalar("LR", float(diffusion.scheduler.optimizer.param_groups[0]["lr"]), current_step)
                # log
                if current_step % opt["train"]["print_freq"] == 0:
                    logs = diffusion.get_current_log()
                    lr = diffusion.scheduler.optimizer.param_groups[0]["lr"]
                    message = f"<epoch:{current_epoch:3d}, iter:{current_step:8,d}, lr:{lr:8,f}> "
                    for k, v in logs.items():
                        message += f"{k:s}: {v:.4e} "
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
                    # if current_step % opt['train']['save_sample_freq'] == 0:
                    #     diffusion.test(continous=False)
                    # output_sr = diffusion.get_current_visuals()['SR']
                    # input_hr = diffusion.get_current_visuals()['HR']
                    # fake_hr = diffusion.get_current_visuals()['INF']
                    # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/image_prediction"+str(current_step)+".tif", np.asarray(output_sr))
                    # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/image_input"+str(current_step)+".tif", np.asarray(fake_hr))
                    # tifffile.imwrite("/home/pd/lucham001/Projects/Image-Super-Resolution-via-Iterative-Refinement/results/image_hr"+str(current_step)+".tif", np.asarray(input_hr))
                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt["train"]["val_freq"] == 0:
                    avg_psnr = 0.0
                    lpips_score = 0.0
                    clip_score = 0.0
                    idx = 0
                    result_path = "{}/{}".format(opt["path"]["results"], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
                    for _, val_data in enumerate(val_loader):
                        idx += 1

                        # print("Index: ", val_data["Index"])
                        diffusion.feed_data(val_data)

                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        # print("SR: ",visuals['SR'].shape)
                        # print("HR: ",visuals['HR'].shape)
                        #
                        #
                        #
                        # sr_img = adaptive_instance_normalization(visuals['SR'][-390:,-390:,:], visuals['LR'][-390:,-390:,:])

                        sr_img = Metrics.tensor2img(sr_img)  # visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals["HR"])  # uint8
                        lr_img = Metrics.tensor2img(visuals["LR"])  # uint8
                        fake_img = Metrics.tensor2img(visuals["INF"])  # uint8

                        hr_img = hr_img[-390:, -390:, :]
                        sr_img = sr_img[-390:, -390:, :]
                        fake_img = fake_img[-390:, -390:, :]

                        # sr_img = adaptive_instance_normalization(fake_img, lr_img)
                        print("shapes: ", hr_img.shape, sr_img.shape)

                        hr_img = np.transpose(hr_img, axes=(1, 2, 0))
                        sr_img = np.transpose(sr_img, axes=(1, 2, 0))
                        fake_img = np.transpose(fake_img, axes=(1, 2, 0))
                        # generation
                        # Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(
                        #     lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(
                        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        plot_img_0 = np.concatenate(
                            (
                                255
                                * (fake_img[:, :, 0] - fake_img[:, :, 0].min())
                                / (fake_img[:, :, 0].max() - fake_img[:, :, 0].min()),
                                255
                                * (sr_img[:, :, 0] - sr_img[:, :, 0].min())
                                / (sr_img[:, :, 0].max() - sr_img[:, :, 0].min()),
                                255
                                * (hr_img[:, :, 0] - hr_img[:, :, 0].min())
                                / (hr_img[:, :, 0].max() - hr_img[:, :, 0].min()),
                            ),
                            axis=1,
                        )
                        plot_img_1 = np.concatenate(
                            (
                                255
                                * (fake_img[:, :, 1] - fake_img[:, :, 1].min())
                                / (fake_img[:, :, 1].max() - fake_img[:, :, 1].min()),
                                255
                                * (sr_img[:, :, 1] - sr_img[:, :, 1].min())
                                / (sr_img[:, :, 1].max() - sr_img[:, :, 1].min()),
                                255
                                * (hr_img[:, :, 1] - hr_img[:, :, 1].min())
                                / (hr_img[:, :, 1].max() - hr_img[:, :, 1].min()),
                            ),
                            axis=1,
                        )
                        plot_img_2 = np.concatenate(
                            (
                                255
                                * (fake_img[:, :, 2] - fake_img[:, :, 2].min())
                                / (fake_img[:, :, 2].max() - fake_img[:, :, 2].min()),
                                255
                                * (sr_img[:, :, 2] - sr_img[:, :, 2].min())
                                / (sr_img[:, :, 2].max() - sr_img[:, :, 2].min()),
                                255
                                * (hr_img[:, :, 2] - hr_img[:, :, 2].min())
                                / (hr_img[:, :, 2].max() - hr_img[:, :, 2].min()),
                            ),
                            axis=1,
                        )
                        plot_img_3 = np.concatenate(
                            (
                                255
                                * (fake_img[:, :, 3] - fake_img[:, :, 3].min())
                                / (fake_img[:, :, 3].max() - fake_img[:, :, 3].min()),
                                255
                                * (sr_img[:, :, 3] - sr_img[:, :, 3].min())
                                / (sr_img[:, :, 3].max() - sr_img[:, :, 3].min()),
                                255
                                * (hr_img[:, :, 3] - hr_img[:, :, 3].min())
                                / (hr_img[:, :, 3].max() - hr_img[:, :, 3].min()),
                            ),
                            axis=1,
                        )

                        plot_img_full = np.expand_dims(
                            np.concatenate((plot_img_0, plot_img_1, plot_img_2, plot_img_3), axis=0), axis=0
                        )
                        plot_img_full = np.transpose(plot_img_full, [0, 2, 1])

                        # print(fake_img.shape, sr_img.shape, lr_img.shape)
                        # print(fake_img[-390:].shape, sr_img.shape, lr_img[-390:].shape)

                        # fake_img = np.expand_dims(fake_img, axis = 0)
                        # hr_img = np.expand_dims(hr_img, axis = 0)
                        # sr_img = np.expand_dims(sr_img, axis = 0)

                        # fake_img = fake_img

                        tb_logger.add_image(
                            f"Iter_{current_step}",
                            plot_img_full,
                            # np.transpose(np.concatenate(
                            #     (255*(fake_img[-390:]-fake_img[-390:].min()) /(fake_img[-390:].max()-fake_img[-390:].min()), 255*(sr_img - sr_img.min())/(sr_img.max()-sr_img.min()), 255*(hr_img[-390:] - hr_img[-390:].min())/(hr_img[-390:].max()-hr_img[-390:].min())), axis=1), [3, 0, 1, 2]),
                            idx,
                        )
                        # dataformats="NCHW")
                        avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)  # [-390:])
                        lpips_score += Metrics.calculate_lpips(sr_img, hr_img, "vgg")
                        clip_score += Metrics.calculate_clipscore(sr_img, hr_img, "clip-ViT-B/16")
                        if wandb_logger:
                            wandb_logger.log_image(
                                f"validation_{idx}", np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    lpips_score = lpips_score / idx
                    clip_score = clip_score / idx

                    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["train"], schedule_phase="train")
                    # log
                    logger.info(
                        f"# Validation # PSNR: {avg_psnr:.4e} # CLIPScore: {clip_score:.4e} # LPIPS: {lpips_score:.4e}"
                    )
                    logger_val = logging.getLogger("val")  # validation logger
                    logger_val.info(f"<epoch:{current_epoch:3d}, iter:{current_step:8,d}> psnr: {avg_psnr:.4e}")

                    if avg_psnr > best_psnr:
                        diffusion.save_network(current_epoch, current_step, best_ckp=True)
                        logger.info(
                            f"# Saving New Best model checkpoint at epoch {current_epoch:3d}, iter {current_step:8,d}"
                        )
                        if os.path.exists(best_checkpoint):
                            os.remove(best_checkpoint)
                        if os.path.exists(best_opt_checkpoint):
                            os.remove(best_opt_checkpoint)
                        best_checkpoint = os.path.join(
                            opt["path"]["checkpoint"], opt["name"] + f"_I{current_step}_E{current_epoch}_best_gen.pth"
                        )
                        best_opt_checkpoint = os.path.join(
                            opt["path"]["checkpoint"], opt["name"] + f"_I{current_step}_E{current_epoch}_best_opt.pth"
                        )
                        best_psnr = avg_psnr

                    # tensorboard logger
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalars(
                        "perceptual-metrics", {"LPIPS": lpips_score, "CLIPScore": clip_score}, current_step
                    )

                    if wandb_logger:
                        wandb_logger.log_metrics({"validation/val_psnr": avg_psnr, "validation/val_step": val_step})
                        val_step += 1

                if current_step % opt["train"]["save_checkpoint_freq"] == 0:
                    logger.info("Saving models and training states.")
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt["log_wandb_ckpt"]:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({"epoch": current_epoch - 1})

        # save model
        logger.info("End of training.")
    else:
        logger.info("Begin Model Evaluation.")
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = "{}".format(opt["path"]["results"])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            print(visuals["LR"].shape, visuals["SR"][-1].shape)
            sr_img = visuals["SR"][-1]
            # sr_img = adaptive_instance_normalization(visuals['SR'][-1], visuals['LR'][0])
            print("SR image shape: ", sr_img.shape)

            hr_img = Metrics.tensor2img(visuals["HR"])  # uint8
            lr_img = Metrics.tensor2img(visuals["LR"])  # uint8
            # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            # hr_img = np.transpose(hr_img[-192:,-192:,:], axes=(2,0,1))
            # lr_img = np.transpose(lr_img[-192:,-192:,:], axes=(2,0,1))
            # fake_img = np.transpose(fake_img[-192:,-192:,:], axes=(2,0,1))

            print(hr_img.shape, lr_img.shape)

            sr_img_mode = "grid"
            if sr_img_mode == "single":
                # single img series
                # sr_img = visuals['SR'][-1]  # uint8

                # sr_img = np.transpose(sr_img, axes=(2,0,1))
                # print("SR Image: ", sr_img.shape)
                # sample_num = sr_img.shape[0]
                # for iter in range(0, sample_num):
                #     Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.tif'.format(result_path, current_step, idx, iter))
                Metrics.save_img(Metrics.tensor2img(sr_img), f"{result_path}/{current_step}_{idx}_sr.tif")

            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals["SR"][-1])  # uint8
                # print("SR Image before:",  sr_img[:,-390:,-390:].shape)
                # sr_img = sr_img[:,-390:,-390:]
                print("SR Image: ", sr_img.shape)

                # sr_img = np.transpose(sr_img, axes=(2,0,1))
                # sr_img = sr_img[-192:,-192:,:]
                # Metrics.save_img(sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(sr_img, f"{result_path}/{current_step}_{idx}_sr.tif")

            print("Saving images: ", f"{result_path}/{current_step}_{idx}_hr.tif")
            Metrics.save_img(hr_img, f"{result_path}/{current_step}_{idx}_hr.tif")
            Metrics.save_img(lr_img, f"{result_path}/{current_step}_{idx}_lr.tif")
            # Metrics.save_img(fake_img, '{}/{}_{}_inf.tif'.format(result_path, current_step, idx))

            # generation
            # eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            # eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            # avg_psnr += eval_psnr
            # avg_ssim += eval_ssim

            # print(eval_psnr, eval_ssim)

            if wandb_logger and opt["log_eval"]:
                wandb_logger.log_eval_data(
                    fake_img, Metrics.tensor2img(visuals["SR"][-1]), hr_img, eval_psnr, eval_ssim
                )

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info(f"# Validation # PSNR: {avg_psnr:.4e}")
        logger.info(f"# Validation # SSIM: {avg_ssim:.4e}")
        logger_val = logging.getLogger("val")  # validation logger
        logger_val.info(
            f"<epoch:{current_epoch:3d}, iter:{current_step:8,d}> psnr: {avg_psnr:.4e}, ssim：{avg_ssim:.4e}"
        )

        if wandb_logger:
            if opt["log_eval"]:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({"PSNR": float(avg_psnr), "SSIM": float(avg_ssim)})
