
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import argparse
import torch.nn.parallel.data_parallel as data_parallel
import numpy as np
import torch.optim as optim
import time
sys.path.append(os.path.dirname(sys.path[0]))

from model.phasen import PHASEN as Model
from tools.misc import get_learning_rate, save_checkpoint, reload_for_eval, reload_model
from tools.time_dataset import make_loader, DataReader

import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

def train(model, args, device, writer):
    print('preparing data...')
    dataloader, dataset = make_loader(
        args.tr_list,
        args.batch_size,
        num_workers=args.num_threads,
            )
    print_freq = 100
    num_batch = len(dataloader)
    params = model.get_params(args.weight_decay)
    optimizer = optim.Adam(params, lr=args.learn_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=1, verbose=True)
    
    if args.retrain:
        start_epoch, step = reload_model(model, optimizer, args.exp_dir,
                                         args.use_cuda)
    else:
        start_epoch, step = 0, 0
    print('---------PRERUN-----------')
    lr = get_learning_rate(optimizer)
    print('(Initialization)')
    val_loss, val_sisnr = validation(model, args, lr, -1, device)
    writer.add_scalar('Loss/Train', val_loss, step)
    writer.add_scalar('Loss/Cross-Validation', val_loss, step)
    
    writer.add_scalar('SiSNR/Train', -val_sisnr, step)
    writer.add_scalar('SiSNR/Cross-Validation', -val_sisnr, step)

    for epoch in range(start_epoch, args.max_epoch):
        torch.manual_seed(args.seed + epoch)
        if args.use_cuda:
            torch.cuda.manual_seed(args.seed + epoch)
        model.train()
        sisnr_total = 0.0
        sisnr_print = 0.0
        mix_loss_total = 0.0 
        mix_loss_print = 0.0 
        amp_loss_total = 0.0 
        amp_loss_print = 0.0
        phase_loss_total = 0.0
        phase_loss_print = 0.0

        stime = time.time()
        lr = get_learning_rate(optimizer)
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs, wav = data_parallel(model, (inputs,))
            loss = model.loss(outputs, labels, mode='Mix')
            loss[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            step += 1
            sisnr = model.loss(wav, labels, mode='SiSNR')
            
            mix_loss_total += loss[0].data.cpu()
            mix_loss_print += loss[0].data.cpu()
            
            amp_loss_total += loss[1].data.cpu()
            amp_loss_print += loss[1].data.cpu()
            
            phase_loss_total += loss[2].data.cpu()
            phase_loss_print += loss[2].data.cpu()
            
            sisnr_print += sisnr.data.cpu()
            sisnr_total += sisnr.data.cpu()

            del outputs, labels, inputs, loss, wav
            if (idx+1) % 1000 == 0:
                save_checkpoint(model, optimizer, -1, step, args.exp_dir)
            if (idx + 1) % print_freq == 0:
                eplashed = time.time() - stime
                speed_avg = eplashed / (idx+1)
                mix_loss_print_avg = mix_loss_print / print_freq
                amp_loss_print_avg = amp_loss_print / print_freq
                phase_loss_print_avg = phase_loss_print / print_freq
                sisnr_print_avg = sisnr_print / print_freq
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} | lr {:1.4e} |'
                      '{:2.3f}s/batches '
                      '| Mixloss {:2.4f}'
                      '| AMPloss {:2.4f}'
                      '| Phaseloss {:2.4f}'
                      '| SiSNR {:2.4f}'
                      .format(
                          epoch, args.max_epoch, idx + 1, num_batch, lr,
                          speed_avg, 
                          mix_loss_print_avg,
                          amp_loss_print_avg,
                          phase_loss_print_avg,
                          -sisnr_print_avg
                    ))
                sys.stdout.flush()
                writer.add_scalar('SiSNR/Train', -sisnr_print_avg, step)
                writer.add_scalar('Loss/Train', mix_loss_print_avg, step)
                mix_loss_print = 0. 
                amp_loss_print = 0.
                phase_loss_print = 0. 
                sisnr_print = 0.

        eplashed = time.time() - stime
        mix_loss_total_avg = mix_loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
        print(
            'Training AVG.LOSS |'
            ' Epoch {:3d}/{:3d} | lr {:1.4e} |'
            ' {:2.3f}s/batch | time {:3.2f}mins |'
            ' Mixloss {:2.4f}'
            ' SiSNR {:2.4f}'
            .format(
                                    epoch + 1, args.max_epoch,
                                    lr,
                                    eplashed/num_batch,
                                    eplashed/60.0,
                                    mix_loss_total_avg,
                                    -sisnr_total_avg
                ))
        val_loss, val_sisnr = validation(model, args, lr, epoch, device)
        writer.add_scalar('Loss/Cross-Validation', val_loss, step)
        writer.add_scalar('SiSNR/Cross-Validation', -val_sisnr, step)
        writer.add_scalar('learn_rate', lr, step) 
        if val_loss > scheduler.best:
            print('Rejected !!! The best is {:2.6f}'.format(scheduler.best))
        else:
            save_checkpoint(model, optimizer, epoch + 1, step, args.exp_dir, mode='best_model')
        scheduler.step(val_loss)
        sys.stdout.flush()
        stime = time.time()


def validation(model, args, lr, epoch, device):
    dataloader, dataset = make_loader(
            args.cv_list,
            args.batch_size,
            num_workers=args.num_threads,
        )
    model.eval()
    loss_total = 0. 
    sisnr_total = 0.
    num_batch = len(dataloader)
    stime = time.time()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, wav = data_parallel(model, (inputs, ))
            loss = model.loss(outputs, labels,mode='Mix')[0]
            sisnr = model.loss(wav, labels, mode='SiSNR')
            loss_total += loss.data.cpu()
            sisnr_total += sisnr.data.cpu()
            del loss, data, inputs, labels, wav, outputs
        etime = time.time()
        eplashed = (etime - stime) / num_batch
        loss_total_avg = loss_total / num_batch
        sisnr_total_avg = sisnr_total / num_batch
    print('CROSSVAL AVG.LOSS | Epoch {:3d}/{:3d} '
          '| lr {:.4e} | {:2.3f}s/batch| time {:2.1f}mins '
          '| Mixloss {:2.4f} | SiSNR {:2.4f}'.format(epoch + 1, args.max_epoch, lr, eplashed,
                                  (etime - stime)/60.0, loss_total_avg.item(), -sisnr_total_avg.item()))
    sys.stdout.flush()
    return loss_total_avg, sisnr_total_avg


def decode(model, args, device):
    model.eval()

    # If set true, there will be impulse noise 
    # in the border on segements.
    # If you don't care it, you can set True
    decode_do_segement=False
    with torch.no_grad():
        
        data_reader = DataReader(
                args.tt_list,
            )
        output_wave_dir = os.path.join(args.exp_dir, 'rec_wav/')
        if not os.path.isdir(output_wave_dir):
            os.mkdir(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            inputs, utt_id, nsamples = data_reader[idx]
            
            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            window = int(args.sample_rate*6) # 4s

            b,t = inputs.size()
            if t > int(1.5*window) and decode_do_segement:
                outputs = np.zeros(t)
                stride = window//2
                current_idx = 0
                while current_idx + window < t:
                        tmp_input = inputs[:,current_idx:current_idx+window]
                        tmp_output = model(tmp_input)[1][0].cpu().numpy()
                        outputs[current_idx:current_idx+window] += tmp_output 
                        current_idx += stride
                if current_idx < t:
                    tmp_input = inputs[:,current_idx:current_idx+window]
                    tmp_output = model(tmp_input)[1][0].cpu().numpy()
                    outputs[current_idx:current_idx+tmp_output.shape[0]] += tmp_output 
                outputs[stride:current_idx+stride]/=2
            else:
                outputs = model(inputs)[1][0].cpu().numpy()
            outputs = outputs[:nsamples]
            # this just for plot mask 
            #amp, mask, phase = model(inputs)[2] 
            #np.save(utt_id, [amp.cpu().numpy(), mask.cpu().numpy(), phase.cpu().numpy()]) 
            sf.write(os.path.join(output_wave_dir, utt_id), outputs, args.sample_rate) 

        print('Decode Done!!!')


def main(args):
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    args.sample_rate = {
        '8k':8000,
        '16k':16000,
        '24k':24000,
        '48k':48000,
    }[args.sample_rate]
    model = Model(
        rnn_nums=args.rnn_nums,
        win_len=args.win_len,
        win_inc=args.win_inc,
        fft_len=args.fft_len,
        win_type=args.win_type
    )
    if not args.log_dir:
        writer = SummaryWriter(os.path.join(args.exp_dir, 'tensorboard'))
    else:
        writer = SummaryWriter(args.log_dir)
    model.to(device)
    if not args.decode:
        train(model, FLAGS, device, writer)
    reload_for_eval(model, FLAGS.exp_dir, FLAGS.use_cuda)
    decode(model, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser.add_argument('--decode', type=int, default=0, help='if decode')
    parser.add_argument(
        '--exp-dir',
        dest='exp_dir',
        type=str,
        default='exp/cldnn',
        help='the exp dir')
    parser.add_argument(
        '--tr-list', dest='tr_list', type=str, help='the train data list')
    parser.add_argument(
        '--cv-list',
        dest='cv_list',
        type=str,
        help='the cross-validation data list')
    parser.add_argument(
        '--tt-list', dest='tt_list', type=str, help='the test data list')
    
    parser.add_argument(
        '--learn-rate',
        dest='learn_rate',
        type=float,
        default=0.001,
        help='the learning rate in training')
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=20,
        help='the max epochs')
    
    parser.add_argument(
        '--rnn-nums',
        dest='rnn_nums',
        type=int,
        default=300,
        help='the num of rnns ')

    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        help='the batch size in train')
    parser.add_argument(
        '--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument(
        '--seed', dest='seed', type=int, default=20, help='the random seed')
    parser.add_argument(
        '--log-dir',
        dest='log_dir',
        type=str,
        default=None,
        help='the random seed')
    parser.add_argument(
        '--num-threads', dest='num_threads', type=int, default=10)
    parser.add_argument(
        '--window-len',
        dest='win_len',
        type=int,
        default=400,
        help='the window-len in enframe')
    parser.add_argument(
        '--window-inc',
        dest='win_inc',
        type=int,
        default=100,
        help='the window include in enframe')
    parser.add_argument(
        '--fft-len',
        dest='fft_len',
        type=int,
        default=512,
        help='the fft length when in extract feature')
    parser.add_argument(
        '--window-type',
        dest='win_type',
        type=str,
        default='hamming',
        help='the window type in enframe, include hamming and None')
    
    parser.add_argument(
        '--num-gpu',
        dest='num_gpu',
        type=int,
        default=1,
        help='the num gpus to use')
   
    parser.add_argument(
        '--weight-decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument(
        '--clip-grad-norm', dest='clip_grad_norm', type=float, default=10.)
    
    parser.add_argument(
        '--sample-rate', dest='sample_rate', type=str, default='16k')
    parser.add_argument('--retrain', dest='retrain', type=int, default=0)
   
    FLAGS, _ = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    #torch.backends.cudnn.benchmark = True
    if FLAGS.use_cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__dict__)
    print(FLAGS.win_type)
    main(FLAGS)
