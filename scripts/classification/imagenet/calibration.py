# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import math
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon, nd, image
from mxnet.gluon.nn import Block, HybridBlock
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils.quantization import *
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str_array
import ctypes

def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)

def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model with Intel MKL-DNN support')
    parser.add_argument('--model-prefix', type=str, default='ssd_300_vgg16_atrous_voc',
                        help='Prefix of converted model.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='Imagenet directory for validation.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-calib-batch', type=int, default=5)
    parser.add_argument('--data-shape', type=int, default=224,
                        help="Input data shape")
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='The ratio for crop and input size, for validation dataset only')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['int8', 'uint8', 'auto'],
                        help='quantization destination data type for input data')
    parser.add_argument('--enable-calib-quantize', type=bool, default=True,
                        help='If enabled, the quantize op will '
                             'be calibrated offline if calibration mode is '
                             'enabled')
    args = parser.parse_args()
    ctx = mx.cpu(0)
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    logger.info('load model %s' % args.model_prefix)
    calib_mode = args.calib_mode
    logger.info('calibration mode set to %s' % calib_mode)

    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, 0)
    data_shape = [args.batch_size, 3, args.data_shape, args.data_shape]

    sym = sym.get_backend_symbol('MKLDNN')

    # get batch size
    batch_size = args.batch_size
    num_calib_batch = args.num_calib_batch
    logger.info('batch size = %d for calibration' % batch_size)
    logger.info('sampling %d batches for calibration' % num_calib_batch)

    calib_layer = lambda name: name.endswith('_output')
    excluded_sym_names = []

    if calib_mode == 'none':
        raise ValueError('Not Support')
    else:
        logger.info('Creating GluonDataloader for reading detection dataset')
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        """
        Aligning with TF implementation, the default crop-input
        ratio set as 0.875; Set the crop as ceil(input-size/ratio)
        """
        crop_ratio = args.crop_ratio if args.crop_ratio > 0 else 0.875
        resize = int(math.ceil(args.data_shape/crop_ratio))

        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(args.data_shape),
            transforms.ToTensor(),
            normalize
        ])

        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(args.data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params, 
                                                        data_shape=data_shape,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=val_data,
                                                        num_calib_batch=num_calib_batch,
                                                        calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                        logger=logger)
        if calib_mode == 'entropy':
            suffix = '-quantized'
        elif calib_mode == 'naive':
            suffix = '-quantized'
        else:
            raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                             % calib_mode)
        sym_name = '%s-symbol.json' % (args.model_prefix + suffix)
    qsym = qsym.get_backend_symbol('MKLDNN_POST_QUANTIZE')
    save_symbol(sym_name, qsym, logger)
    param_name = '%s-0000.params' % (args.model_prefix + '-quantized')
    save_params(param_name, qarg_params, aux_params, logger)
    # graph = mx.viz.plot_network(qsym)
    # graph.format = 'png'
    # graph.render(args.model_prefix + suffix)
