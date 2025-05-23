import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import torchvision
import os
import numpy as np
import torch_fft_package as fft_mri

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_32_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    best_psnr = 0.0
    best_ssim = 0.0

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                        concatenated_images = np.concatenate((fake_img, sr_img, hr_img), axis=1)
                        final_image = np.expand_dims(concatenated_images, axis=0)
                        tb_logger.add_image('Iter_{}'.format(current_step), final_image, idx)
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)
                        avg_ssim += Metrics.calculate_ssim(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)

                    # Save checkpoint if validation performance improves
                    if avg_psnr >= best_psnr:
                        logger.info('PSNR, Having a best model. Saving models and training states.')
                        diffusion.save_network(0, 0)    # best model name is I0_E0
                        best_psnr = avg_psnr

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

                    if avg_ssim >= best_ssim:
                        logger.info('SSIM, Having a best model. Saving models and training states.')
                        diffusion.save_network(1, 1)    # best model name is I1_E1
                        best_ssim = avg_ssim

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        num_cascades = 1
        undersample_ratio = 1/0.35  # 30% LR

        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1

            acquired_data = val_data['SR'].squeeze()
            kspace_acquired = fft_mri.fft2(acquired_data)

            # Generate and apply the central k-space mask
            mask = fft_mri.create_center_mask(undersample_ratio=undersample_ratio)
            central_kspace_lr = kspace_acquired * mask

            current_sr = val_data['SR']

            hr_path = val_data['Path'][0]
            filename = os.path.basename(hr_path)
            filename = os.path.splitext(filename)[0]

            del val_data['Path']
            logger.info('# Validation filename: {}'.format(filename))

            for cascade in range(num_cascades):
                # Feed data for each cascade iteration
                val_data['SR'] = current_sr
                diffusion.feed_data(val_data)
                diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()

                # Update SR for next cascade
                current_sr = visuals['SR'][-1]

                # Fourier Transform of the new SR
                kspace_sr = fft_mri.fft2(current_sr)
                kspace_sr = kspace_sr * (~mask) + central_kspace_lr
                updated_sr = fft_mri.ifft2(kspace_sr).real
                current_sr = updated_sr.unsqueeze(0)

                # Images are saved at every cascade
                hr_img = Metrics.tensor2img(visuals['HR'])
                lr_img = Metrics.tensor2img(visuals['LR'])
                sr_img = Metrics.tensor2img(updated_sr)  # visuals['SR'][-1] -> updated_sr, one more DC
                fake_img = Metrics.tensor2img(visuals['INF'])
                if cascade == 0:
                    Metrics.save_img(hr_img, '{}/{}_{}_{}_hr.png'.format(result_path, str(idx).zfill(4), filename, cascade + 1))
                    Metrics.save_img(lr_img, '{}/{}_{}_{}_lr.png'.format(result_path, str(idx).zfill(4), filename, cascade + 1))
                Metrics.save_img(sr_img, '{}/{}_{}_{}_sr.png'.format(result_path, str(idx).zfill(4), filename, cascade + 1))
                # Metrics.save_img(fake_img, '{}/{}_{}_{}_inf.png'.format(result_path, str(idx).zfill(4), filename, cascade + 1))

                # Calculate metrics for the final SR result after the last cascade
                if cascade == num_cascades - 1:
                    eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
                    eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)

                    avg_psnr += eval_psnr
                    avg_ssim += eval_ssim

                    # Log evaluation data to Wandb or other platforms if necessary
                    if wandb_logger and opt['log_eval']:
                        wandb_logger.log_eval_data(fake_img, sr_img, hr_img, eval_psnr, eval_ssim)

        # Calculate average metrics over all images
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
