#! /usr/bin/env python

'''
Usage:  Pull all waveforms from wfdisc or directory and build detection tables.
        Detections for P and S waves utilize a Convolutional Encoder-Decoder
        P-wave polarity is estimate as Up/Down/Null

Author: Christopher W Johnson 
        EES-17
        Los Alamos National Laboratory

This program is open source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3.Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import argparse
import configparser
import os, sys, copy, re, warnings, glob
import time
import logging
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import joblib
import random

warnings.simplefilter("ignore")

# Placeholder
# Antelope not functional
# try:
#     import antelope.elog  as elog
#     from antelope.stock     import *
#     from antelope.datascope import *
#     antelope_import = True
# except ImportError:
#     print('Antelope modelues not available')
#     antelope_import = False

try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    sys.exit("Import Error! Tensorflow modules not available. EXIT NOW!")

import utils

# Set seeds for consistent model outputs 
# This also requires setting env var PYTHONHASHSEED=0
np.random.seed(1234)
tf.random.set_seed(1234)
random.seed(1234)


def parse_config_file(fin):
    with open(fin, 'r') as f:
        for line in f:
            # skip the header and input with read_string
            if not line[0] == '#':
                continue
            else:
                fi = f.read()
                break

    config = configparser.ConfigParser(allow_no_value=True)
    config.read_string(fi)
    return config


def setup_logger(name, log_file, level=logging.INFO):
    '''
    Setup multiple loggers
    log file and detection table will have different handle
    '''

    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def create_parser():
    # Parse command line
    description  = 'README: Process phase detections & first motions in fdir. '
    description += 'Expects a directory with mseed files and a '
    description += 'subdirectory called resp/ with xml station response files. '
    description += 'Edit config file for paths to model files.'

    parser = argparse.ArgumentParser(description=description)
    # Positional argument
    parser.add_argument('fdir', type=str, help='wfdisc or mseed directory' )

    parser.add_argument('--label',
                        type=str,
                        default='run1',
                        help='Identifer for output files. Default="run1"')

    parser.add_argument('--config_file',
                        type=str,
                        default='config_file.txt',
                        help='Setup file with model info. Default=config_file.txt')

    parser.add_argument('--dbreader',
                        type=str,
                        default='obspy',
                        help='Backend [antelope or obspy]. Default=obspy')

    parser.add_argument('--gpus',
                        type=int,
                        default=0,
                        help='# GPUs for processing data. If 0, CPU only. Default = 0')

    parser.add_argument('--process',
                        type=int,
                        default=1,
                        help='# process on each gpu, unless gpu=0. Default = 1')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Scale batch size for GPU memory. Default = 16')

    parser.add_argument('-v',
                        '--verbosity',
                        action='count',
                        help='increase output verbosity',
                        default=0)

    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Output test files')

    return parser


def main(args):
    parser = create_parser()
    argv = parser.parse_args(args[1:])

    # Add absolute path
    argv.fdir = os.path.abspath(argv.fdir)

    # Check the inputs and exit if needed
    exit_ = 0
    if not os.path.isdir(argv.fdir) or os.path.isfile(argv.fdir):
        print('\nWfdisc/Directory {0} does not exist\n'.format(argv.fdir))
        exit_ = 1

    if not os.path.isfile(argv.config_file):
        print('\n{0} does not exist\n'.format(argv.config_file))
        exit_ = 1

    if not argv.dbreader.lower() in ['antelope', 'obspy']:
        print('\n{0} not valid dbreader [antelope or obspy]\n'.format(argv.dbreader))
        exit_ = 1

    # Parse config file and command line inputs into params dict
    cf = parse_config_file(argv.config_file)
    # print(cf._sections)
    params = cf._sections['params']
    for key, val in params.items():
        params[key] = float(val)

    for key, val in vars(argv).items():
        params[key] = val

    # Open log file, remove old file if existing
    logfile = f"EQ_phase_detection_{params['label']}.log"
    if os.path.isfile(logfile):
        os.remove(logfile)
    logger = setup_logger('logger', logfile)


    params['detection'] = cf._sections['detection']
    params['polarity']  = cf._sections['polarity']
    params['logger']    = logger



    # Test if Detection model file and dir exists
    d_mod    = os.path.join(params['detection']['path'], params['detection']['model'])
    m_file   = os.path.isfile(d_mod)
    m_dir    = os.path.isdir(d_mod)
    # w_file   = os.path.isfile(d_weight)
    if not (m_file or m_dir):
        print("Detection model does not exist. Check paths in config_file!")
        exit_ = 1

    # Test if Polarity model file or dir exists
    p_mod    = os.path.join(params['polarity']['path'], params['polarity']['model'])
    m_file   = os.path.isfile(p_mod)
    m_dir    = os.path.isdir(p_mod)

    if not (m_file or m_dir):
        print('Polarity model does not exist. Check paths in config_file!')
        exit_ = 1

    if exit_ == 1:
        parser.print_help()
        sys.exit()

    # Record config file
    logger.info('Inputs      - args')
    logger.info('fdir        - {0}'.format(params['fdir']))
    logger.info('config_file - {0}'.format(params['config_file']))
    logger.info('dbreader    - {0}'.format(params['dbreader']))
    logger.info('process     - {0}'.format(params['process']))
    logger.info('gpus        - {0}'.format(params['gpus']))
    logger.info('verbosity   - {0}'.format(params['verbosity']))
    logger.info('\nConfig File')
    logger.info('[params]')
    logger.info('sps         - {0}'.format(params['sps'    ]))
    logger.info('eqdet       - {0}'.format(params['eqdet'  ]))
    logger.info('pwave       - {0}'.format(params['pwave'  ]))
    logger.info('swave       - {0}'.format(params['swave'  ]))
    logger.info('freqmin     - {0}'.format(params['freqmin']))
    logger.info('freqmax     - {0}'.format(params['freqmax']))
    logger.info('\n[detection]')
    logger.info('path        - {0}'.format(params['detection']['path']))
    logger.info('npts        - {0}'.format(params['detection']['npts']))
    # logger.info('nperseg     - {0}'.format(params['detection']['nperseg']))
    # logger.info('noverlap    - {0}'.format(params['detection']['noverlap']))
    logger.info('model       - {0}'.format(params['detection']['model']))
    # logger.info('weights     - {0}'.format(params['detection']['weights']))
    logger.info('\n[polarity]')
    logger.info('path        - {0}'.format(params['polarity']['path']))
    logger.info('npts        - {0}'.format(params['polarity']['npts']))
    logger.info('model       - {0}'.format(params['polarity']['model']))
    # logger.info('weights     - {0}'.format(params['polarity']['weights']))

    if params['dbreader'] == 'obspy':
        logger.info('\nDetection file columns')
        logger.info('net stat phs  utctime polarity PGD SNR amp(mm) prob(softmax)')
    logger.info('\n---------------Processing---------------\n')

    tic  = datetime.datetime.now()
    # Build Scheduler with models loaded on each gpu
    ids = []
    if params['gpus'] == 0:
        for ci in range(params['process']):
            ids.append(['NA', ci])
    else:
        for gi in range(params['gpus']):
            for ci in range(params['process']):
                ids.append([gi, ci])
    params['ids'] = ids
    det_fin = 0

    # Obspy
    if params['dbreader'].lower() == 'obspy':
        # Obspy Detection File
        detectfile = f"EQ_phase_detection_{params['label']}.detection"
        try:
            os.remove(detectfile)
        except(FileNotFoundError):
            print('')
        params['detect'] = setup_logger('detect', detectfile)

        # list of files to process
        ##########
        # This is hard coded to mseed becase the obspy reader is setup for this format.
        # If could change to sac, which is 1 file per channel, or other formats
        # that are supported by obspy. However, mseed format is downloaded from 
        # data centers and provides the best file compression
        #
        # The expectation is a /resp file with coresponding xml response files
        ##########
        
        # Process
        daily_stations = sorted(glob.glob(os.path.join(params['fdir'], '*.mseed')))
        
        # classWithModels = utils.detection_utils.Scheduler(**params)
        classWithModels = utils.detection_utils.Scheduler(**params)
        det_fin = classWithModels.start(iter(daily_stations))


    # Processing Complete
    if det_fin:
        # Merge dataframes with pwave and first motions
        f_df = glob.glob('tmp_df/P*.pkl')
        df = joblib.load(f_df[0])
        for f_df_ in f_df[1:]:
            df = df.append(joblib.load(f_df_))
        df.index = np.arange(df.index.size)
        dffile = 'EQ_phase_detection_{0}.df.pkl'.format(params['label'])
        joblib.dump(df, dffile)
        # remove tmp files/dir
        for f_df_ in f_df:
            os.remove(f_df_)
        os.rmdir('tmp_df')
    else:
        print('Error in processing. Did not finish and dataframe is in tmp_df')

    toc      = datetime.datetime.now()
    duration = relativedelta(toc, tic)
    str_prt  = 'Runtime {days:02d}-{hours:02d}:{mins:02d}:{secs:02d}'
    str_prt  = str_prt.format(days=duration.days, 
                              hours=duration.hours, 
                              mins=duration.minutes, 
                              secs=duration.seconds)
    print(str_prt)
    logging.info(str_prt)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(['-h'])
    main(sys.argv)
