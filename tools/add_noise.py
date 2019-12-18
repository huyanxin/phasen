import sys
import os

import scipy.io as sio
import scipy
import numpy as np
import multiprocessing
import wave
import argparse 
import soundfile as sf

os.environ['OMP_NUM_THREADS'] = '2'

def activelev(data):
    '''
        need to update like matlab
    '''
    max_amp = np.std(data)#np.max(np.abs(data))
    return data/max_amp

def add_noisem(clean_path, noise_path, out_clean_dir, out_noisy_dir, start, scale, snr, mode='train'):

    try:    
        clean = read(clean_path)
        noise = read(noise_path)
        cname = clean_path.split('/')[-1].split('.wav')[0]
        nname = noise_path.split('/')[-1].split('.wav')[0]
        if mode != 'test':
            name = cname+'_'+str(snr)+'_'+nname+'_'+str(-snr)+'.wav'
        else:
            name = cname+'.wav'
        clean_size = clean.shape[0]
        if start < 0:
            noise_selected = np.concatenate([noise,noise[1:]-0.97*noise[:-1]])[:clean_size]
        
        else: 
            noise_selected = noise[start:start+clean_size]
        clean_n = activelev(clean)
        noise_n = activelev(noise_selected)
        clean_snr = snr/2.
        noise_snr = -snr/2.
        #clean_weight = 1. #10**(clean_snr/20)
        #noise_weight = np.sqrt(np.var(clean_n)/np.var(noise_n)/10**(snr/10))
        clean_weight = 10**(clean_snr/20)
        noise_weight = 10**(noise_snr/20)
        clean = clean_n * clean_weight
        noise = noise_n * noise_weight
        noisy = clean + noise
        max_amp = np.max(np.abs([noise, clean, noisy]))
        mix_scale = 1/max_amp*scale
        X = clean * mix_scale
        Y = noisy * mix_scale
        write(out_clean_dir+'/'+name, X)
        write(out_noisy_dir+'/'+name, Y)
    except Exception as e :
        print(e)

def read(path):
    """
        read wave data like matlab's audioread
    """
    return sf.read(path)[0]#np.reshape(wavedata, [-1, nchannels])[:, 0]

def write(path, data):
    sf.write(path,data,16000)

def AddNoise(mix_list, out_clean_dir, out_noisy_dir, num_threads=14):
    mode = None
    if 'test' in mix_list or 'tt' in mix_list:
        mode = 'test'
    pool = multiprocessing.Pool(num_threads)
    if not os.path.isdir(out_clean_dir):
        os.mkdir(out_clean_dir)
    if not os.path.isdir(out_noisy_dir):
        os.mkdir(out_noisy_dir)
    with open(mix_list) as fid:
        for line in fid:
            tmp = line.strip().split()
            cname, nname, start, snr, scale = tmp
            start = int(start)
            scale = float(scale)
            snr = float(snr)
            pool.apply_async(
                add_noisem, args=(
                            cname,
                            nname,
                            out_clean_dir,
                            out_noisy_dir, 
                            start,
                            scale,
                            snr,
                            mode                     
                )
            )
            
    pool.close()
    pool.join()
    

def generate_mix_list(cwav_list, nwav_list, output_list, snr_range=[-5,5]):
    '''
        cwav_list: include clean wav path list
        nwav_list: include noise wav path list
        output_list: output cwav path, nwav_path, start_time, scale snr
    '''
    noise_lists = []
    
    with open(nwav_list) as nfid:
        for line in nfid:
            noise_lists.append(line.strip().split()[0])
    
    noise_lists_length = len(noise_lists)
    with open(cwav_list) as cfid:
        with open(output_list, 'w') as outfid:
            for line in cfid:
                cpath = line.strip().split()[0]
                cwav_len = read(cpath).shape[0]
                while True:
                    nid = np.random.randint(noise_lists_length)
                    nwav_len = read(noise_lists[nid]).shape[0]
                    if nwav_len < cwav_len//2:
                        continue
                    else:
                        break
                if cwav_len < nwav_len:
                    stime = np.random.randint(nwav_len-cwav_len)
                elif cwav_len == nwav_len:
                    stime = 0
                elif cwav_len > nwav_len:
                    stime = -1

                if isinstance(snr_range, list):
                    snr = np.random.uniform(snr_range[0], snr_range[1])
                else:
                    snr = snr_range
                t = np.random.normal() * 0.5 + 0.9
                lower=0.3
                upper=0.99
                if t < lower or t > upper:
                    t = np.random.uniform(lower, upper) 
                scale = t
                outfid.writelines(cpath+' '+ noise_lists[nid] + ' ' + str(stime)+' {:.3f}'.format(snr)+' {:.3f}'.format(scale)+'\n')
                sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clean_wav_list',
        type=str,
        default='clean.lst'
    )
    parser.add_argument(
        '--noise_wav_list',
        type=str,
        default='noise.lst'
    )
    parser.add_argument(
        '--mix_list',
        type=str,
        default='mix.lst'
    )
    parser.add_argument(
        '--generate_mix_wav',
        type=int,
        default=0
    )
    parser.add_argument(
        '--snr_lower',
        type=int,
        default=-5
    )
    parser.add_argument(
        '--snr_upper',
        type=int,
        default=20
    )
    
    parser.add_argument(
        '--output_clean_dir',
        type=str,
        default='../data/wavs/clean'
    )
    parser.add_argument(
        '--output_noisy_dir',
        type=str,
        default='../data/wavs/noisy'
    )

    args = parser.parse_args() 
    name = args.mix_list
    clean = args.clean_wav_list
    noise = args.noise_wav_list
    if args.snr_lower == args.snr_upper:
        snr = args.snr_lower
    else:
        snr = [args.snr_lower, args.snr_upper]
    if not os.path.isfile(name):
        generate_mix_list(clean , noise, name, snr_range=snr)
    print('generated mix list')
    if args.generate_mix_wav:
        AddNoise(name, args.output_clean_dir, args.output_noisy_dir, num_threads=12)

