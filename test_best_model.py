'''
Evaluation function
'''

import argparse
import time

import matplotlib.pyplot as plt
import neptune

import os, sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_data import load_data, min_max_norm, get_phase_mag
from model.dvae import *
from train_hyperopt import TimeSpecConverter

from utils.utils_vis import plt_ori_vs_gen, plt_ori_vs_rec
from metrics.discrimanitive import discriminative_score_metrics
from metrics.visualization_metrics import visualization

import logging
import torchaudio

# hyperopt dependencies
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import functools
import numpy as np
import json

import re
from gms_util import get_gms_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_metrics(args, model, test_loader, all_set, all_loader, run, time_spec_converter, norm_dict, SEQ_LEN):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model.eval()
    wfs_min, wfs_max = norm_dict['log_wfs']
    
    with torch.no_grad():
        real_wfs_list, pred_wfs_list = [], []

        real_wfs_list, pred_wfs_list = collect_real_and_gen(args, model, all_loader, pred_wfs_list, real_wfs_list, SEQ_LEN)
        true_phase_list = all_set[2]

        real_wfs_list = np.transpose(np.concatenate(tuple(real_wfs_list), axis=0), (0, 2, 1))
        pred_wfs_list = np.transpose(np.concatenate(tuple(pred_wfs_list), axis=0), (0, 2, 1))

        real_wfs_list = time_spec_converter.time_to_spec(torch.Tensor(real_wfs_list).squeeze().to(args.device)).permute(0, 2, 1).detach().cpu()
        _, real_wfs_list = get_phase_mag(real_wfs_list)
        pred_wfs_list = np.transpose(pred_wfs_list, (0, 2, 1))

        # get the reconstruction and posterior
        test_iter = iter(test_loader)
        wfs, cond_var, true_phase , _  = next(test_iter)

        cond_var = cond_var.to(args.device).float()
        wfs_hat, _, _ = model(wfs, cond_var)

        wfs = min_max_norm(wfs, wfs_min, wfs_max, '[0,1]', 'add')
        wfs_hat = min_max_norm(wfs_hat, wfs_min, wfs_max, '[0,1]', 'add')

        eps = 1e-10
        
        # transform amplitude to original waveforms
        wfs_hat = torch.pow(10, wfs_hat) - eps
        wfs = torch.pow(10, wfs) - eps
        
        conv_input_orig = wfs.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase.to(args.device))
        conv_input_hat  = wfs_hat.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase.to(args.device))

        wfs_orig = time_spec_converter.spec_to_time(conv_input_orig).unsqueeze(dim=-1).detach().cpu()
        wfs_hat = time_spec_converter.spec_to_time(conv_input_hat).unsqueeze(dim=-1).detach().cpu()

        pred_wfs_list = torch.Tensor(pred_wfs_list)
        pred_wfs_list = min_max_norm(pred_wfs_list, wfs_min, wfs_max, '[0,1]', 'add')
        pred_wfs_list = torch.pow(10, pred_wfs_list) - eps
        
        wfs_gen = time_spec_converter.spec_to_time(pred_wfs_list.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase_list.to(args.device))).unsqueeze(dim=-1).detach().cpu()
        wfs_all_orig = time_spec_converter.spec_to_time(real_wfs_list.permute(0, 2, 1).to(args.device)*torch.exp(1j * true_phase_list.to(args.device))).unsqueeze(dim=-1).detach().cpu()

        visualization(real_wfs_list, pred_wfs_list, 'tsne', args, run)
        plt_ori_vs_rec(wfs_orig, wfs_hat, run)

        test_loader_iter = iter(test_loader)
        _, cond_var, true_phase, wfs = next(test_loader_iter)
        
        # [b,c,h,w]
        wfs = wfs.to(args.device).float()
        cond_var = cond_var.to(args.device).float()

        pred_wfs = model.module.generate(cond_var, SEQ_LEN)  # [679,3,6000]
        pred_wfs = min_max_norm(pred_wfs, wfs_min, wfs_max, '[0,1]', 'add')

        # transform stft wfs to original signal
        pred_wfs = torch.pow(10, pred_wfs) - eps

        pred_wfs = time_spec_converter.griffinlim(pred_wfs.permute(0, 2, 1)).detach().cpu()
        # pred_wfs = time_spec_converter.spec_to_time(pred_wfs.permute(0, 2, 1)*torch.exp(1j * true_phase.to(args.device))).unsqueeze(dim=-1).detach().cpu()

        wfs, pred_wfs = wfs.cpu().numpy(), pred_wfs.cpu().numpy()

        for i in range(wfs.shape[0]):
            f, ax = plt.subplots(1)
            plt.plot(wfs[i].squeeze(), label='original')
            plt.plot(pred_wfs[i].squeeze(), label='generated')
            plt.legend()
            # run['test/gen_con_var'].log(f)
            plt.savefig(get_gms_path(f'figures/wfs{i}_{args.loc}.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(f)

            if i == 20:
                break

    discriminative_score = list()
    for _ in range(10):
        disc_score = discriminative_score_metrics(wfs_all_orig, wfs_gen, args)
        discriminative_score.append(disc_score)
        print(disc_score)

    print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)) + 
          ', std: ' + str(np.round(np.std(discriminative_score), 4)))

    return np.round(np.mean(discriminative_score), 4), np.round(np.std(discriminative_score), 4)


def collect_real_and_gen(args, model, loader, pred_wfs_list, real_wfs_list, SEQ_LEN):
    for batch_idx, (_, cond_var, _, wfs) in enumerate(loader):

        # [b,c,h,w]
        wfs = wfs.to(args.device).float()
        cond_var = cond_var.to(args.device).float()

        pred_wfs = model.module.generate(cond_var, SEQ_LEN)  # [679,3,6000]

        # scale wfs back to original magnitude
        wfs, pred_wfs = wfs.cpu().numpy(), pred_wfs.cpu().numpy()

        real_wfs_list.append(wfs)
        pred_wfs_list.append(pred_wfs)

    return real_wfs_list, pred_wfs_list


def save_checkpoint(ckpt_dir, state):
    saved_state = {
    'model': state['model'].state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def restore_checkpoint(ckpt_dir, state, device='cuda:0'):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        return state


def extract_parameter_values(input_string, parameter_names):
    parameter_values = {}
    print("parameter values: ", end="")
    for param in parameter_names:
        pattern = re.compile(f"{param}=(\d+)")
        match = pattern.search(input_string)

        if match:
           parameter_values[param] = int(match.group(1))
           print(f"{param}={parameter_values[param]} ", end="")
    print("")
    return parameter_values


def eval(args):
    checkpoint_file = get_gms_path(args.log_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"checkpoint file not found - {checkpoint_file}")
        
    parameter_names = ["rnn_size", "z_dim", "w_len", "h_len", "tcondvar", "ncond", "bs"]
    parameter_values = extract_parameter_values(checkpoint_file, parameter_names)
        
    fft_size = parameter_values['w_len']
    w_len = parameter_values['w_len']
    h_len = parameter_values['h_len']
    tcondvar = parameter_values['tcondvar']
    batch_size = parameter_values['bs']
    z_dim = parameter_values['z_dim']
    z_rnn_dim = parameter_values['rnn_size']
    ncond = parameter_values['ncond']

    set_seed(args)

    time_spec_converter = TimeSpecConverter(n_fft=fft_size, w_len=w_len, h_len=h_len, power=1, device=args.device)

    _, _, all_set, _, test_loader, all_loader, norm_dict, time_serie_len = load_data(args.path, args.data_file, args.idx_file, time_spec_converter, train_bs=batch_size, tcondvar=tcondvar)

    # setup the model
    model = cVAE(in_dim=fft_size, z_dim=z_dim, ncond=ncond, z_rnn_dim=z_rnn_dim, in_size=len(norm_dict)-1).to(args.device)
    model = torch.nn.DataParallel(model, device_ids=[args.device])

    state = dict(model=model)
    state = restore_checkpoint(checkpoint_file, state, args.device)
    model = state['model']

    SEQ_LEN = time_serie_len//h_len + 1

    run = None

    disc_mean, disc_std = eval_metrics(args, model, test_loader, all_set, all_loader, run, time_spec_converter, norm_dict, SEQ_LEN)
    print('Discriminative score: {:.4f} +- {:.4f}'.format(disc_mean, disc_std))


def main(args, mc=None):
    checkpoint_file = get_gms_path(args.log_dir, args.checkpoint_file)
    if not os.path.exists(get_gms_path(args.log_dir, args.checkpoint_file)):
        raise FileNotFoundError(f"checkpoint file not found - {checkpoint_file}")
    
    parameter_names = ["rnn_size", "z_dim", "w_len", "h_len", "tcondvar", "ncond", "bs"]
    parameter_values = extract_parameter_values(checkpoint_file, parameter_names)
        
    fft_size = parameter_values['w_len']
    w_len = parameter_values['w_len']
    h_len = parameter_values['h_len']
    tcondvar = parameter_values['tcondvar']
    batch_size = parameter_values['bs']
    z_dim = parameter_values['z_dim']
    z_rnn_dim = parameter_values['rnn_size']
    ncond = parameter_values['ncond']
    set_seed(args)
    
    if w_len == 160 and h_len==46:
        # Get data
        print('========================')
        print('Loading data...')
        print('========================\n')

        time_spec_converter = TimeSpecConverter(n_fft=fft_size, w_len=w_len, h_len=h_len, power=1, device=args.device)
        _, _, all_set, train_loader, test_loader, all_loader, norm_dict, time_serie_len = load_data(args.path, args.data_file, args.idx_file, time_spec_converter, train_bs=batch_size, tcondvar=tcondvar)

        # setup the model
        print("Setting up the model...")
        model = cVAE(in_dim=fft_size, z_dim=z_dim, ncond=ncond, z_rnn_dim=z_rnn_dim, in_size=len(norm_dict)-1).to(args.device)
        model = torch.nn.DataParallel(model, device_ids=[args.device])

        print("Restoring the checkpoint ...")
        state = dict(model=model)
        state = restore_checkpoint(checkpoint_file, state, args.device)
        model = state['model']
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        SEQ_LEN = time_serie_len//h_len + 1
        eps = 1e-10
        wfs_min, wfs_max = norm_dict['log_wfs']
        griffinlim = torchaudio.transforms.GriffinLim(n_fft=fft_size, n_iter=500, win_length=w_len, hop_length=h_len, power=1).to(args.device)

        print("Evaluating the model ...")
        model.eval()
        real_wfs_list, pred_wfs_list = [], []
        
        with torch.no_grad():
            for _, (_, cond_var, true_phase, wfs) in enumerate(all_loader):
                # [b,c,h,w]
                wfs = wfs.to(args.device).float()
                cond_var = cond_var.to(args.device).float()
                
                pred_wfs_sub_lst = []
                prev = None
                
                print("Generating random samples ...")
                # generate 100 random samples for each set of conditional variables
                for _ in range(100):
                    print(f"{_}. module generating ..")
                    pred_wfs = model.module.generate(cond_var, SEQ_LEN)
                    pred_wfs = pred_wfs.permute(0, 2, 1)
                    pred_wfs = min_max_norm(pred_wfs, wfs_min, wfs_max, '[0,1]', 'add')
                    pred_wfs = torch.pow(10, pred_wfs) - eps
                    print("running griffinlim ...")
                    pred_wfs = griffinlim(pred_wfs)
                    print("appending to list ...")
                    # pred_wfs = time_spec_converter.spec_to_time(pred_wfs*torch.exp(1j * true_phase.to(args.device))).unsqueeze(dim=-1).detach()
                    pred_wfs_sub_lst.append(pred_wfs.cpu().numpy())
                    # if prev is not None:
                    #     print(np.allclose(pred_wfs.detach().cpu().numpy()), prev)
                    # prev = pred_wfs.detach().cpu().numpy()

                real_wfs_list.append(wfs.cpu().numpy())
                pred_wfs_list.append(pred_wfs_sub_lst)

            print(np.vstack(real_wfs_list).shape)
            print(np.vstack(pred_wfs_list).shape)
            np.save(get_gms_path(f'gens/wfs_original_samples_hlen{h_len}_wlen{w_len}_rnndim{z_rnn_dim}_zdim{z_dim}_{args.loc}_real.npy'), np.transpose(np.vstack(real_wfs_list), (0, 2, 1)))
            np.save(get_gms_path(f'gens/wfs_generated_samples_hlen{h_len}_wlen{w_len}_rnndim{z_rnn_dim}_zdim{z_dim}_{args.loc}_pred.npy'), np.vstack(pred_wfs_list))
    
        
def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--path', type=str, default=get_gms_path('/data'), help='data directory')  
    parser.add_argument('--log_dir', type=str, default=get_gms_path('/log'), help='model saving directory')  
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--z_dim', type=int, default=16, help='number of channels of latent space z')
    parser.add_argument('--z_rnn_dim', type=int, default=32, help='number of channels of latent space z using rnn')
    parser.add_argument('--ncond', type=int, default=16, help='number of channels of conditional variables')
    parser.add_argument('--beta', type=float, default=0.05, help='penalty coefficient for reconstruction loss')
    parser.add_argument('--norm_mode', type=str, default='global', help='normalization method')
    parser.add_argument('--neptune', default='async')
    parser.add_argument('--tag', default='gm vae reg hlen')
    parser.add_argument('--log_reg', action='store_true')

    # audio args
    parser.add_argument('--w_len', type=int, default=160, help='window length')
    parser.add_argument('--h_len', type=int, default=46, help='hop length') 
    parser.add_argument('--power', type=int, default=1, help='power of the spectrogram')
    parser.add_argument('--fft_size', type=int, default=160, help='fft size')

    #custom arguments
    parser.add_argument('--data_file', type=str, dest='data_file', default='data.csv', help='data file name or path')  
    parser.add_argument('--idx_file', type=str, dest='idx_file', default='idx.npy', help='idx file name or path')  
    parser.add_argument('--loc', type=str, dest='loc', default='EW', help='location') 
    parser.add_argument('--checkpoint_file', type=str, dest='checkpoint_file', \
                        default=get_gms_path('GM_V2_VAE_data5_dist-5000_bs=128-rnn_size=32-z_dim=32-lr=0.0006-weight:kl=0.08-log_reg=True-w_decay=1e-06-w_len=160-h_len=46-ncond=32-tcondvar=4-seed=3407'), \
                        help='checkpoint file name')  
    parser.add_argument('--only_eval', dest='only_eval', default=False, action=argparse.BooleanOptionalAction, help='just run eval()')
    parser.add_argument('--only_visualization', dest='only_visualization', default=False, action=argparse.BooleanOptionalAction, help='just visualization')
    parser.add_argument('--real_wfs_file', type=str, dest='real_wfs_file', default='real_wfs_file.csv', help='real wfs file')  
    parser.add_argument('--pred_wfs_file', type=str, dest='pred_wfs_file', default='pred_wfs_file.csv', help='predicted wfs file')  
    
    

    args = parser.parse_args()
    if args.only_eval:
        eval(args)
    elif args.only_visualization:
        if not os.path.exists(get_gms_path(args.real_wfs_file)):
            logging.warning(f"No real wfs file found")
            exit
        if not os.path.exists(get_gms_path(args.pred_wfs_file)):
            logging.warning(f"No pred wfs file found")
            exit
        real_wfs_list = np.load(get_gms_path(args.real_wfs_file))
        pred_wfs_list = np.load(get_gms_path(args.pred_wfs_file))
        run = None
        visualization(real_wfs_list, pred_wfs_list, 'tsne', args, run)
    else:
        main(args)
    
