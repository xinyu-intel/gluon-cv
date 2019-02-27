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
import logging
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from quantization import *
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

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=num_workers)
    return val_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model with Intel MKL-DNN support')
    parser.add_argument('--model-prefix', type=str, default='ssd_300_vgg16_atrous_voc',
                        help='Prefix of converted model.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-calib-batch', type=int, default=5)
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
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
    excluded_sym_names = ['ssd0_flatten0',
                          'ssd0_flatten1',
                          'ssd0_flatten2',
                          'ssd0_flatten3',
                          'ssd0_flatten4',
                          'ssd0_flatten5',
                          'ssd0_flatten6',
                          'ssd0_flatten7',
                          'ssd0_flatten8',
                          'ssd0_flatten9',
                          'ssd0_flatten10',
                          'ssd0_flatten11']

    excluded_sym_names += ['ssd0_concat0',
                            'ssd0_multiperclassdecoder0_concat0',
                            'ssd0_concat1',
                            'ssd0_concat2',
                            'ssd0_normalizedboxcenterdecoder0_concat0',
                            'ssd0_concat3',
                            'ssd0_concat4',
                            'ssd0_concat5',
                            'ssd0_concat6',
                            'ssd0_concat7',
                            'ssd0_concat8',
                            'ssd0_concat9',
                            'ssd0_concat10',
                            'ssd0_concat11',
                            'ssd0_concat12',
                            'ssd0_concat13',
                            'ssd0_concat14',
                            'ssd0_concat15',
                            'ssd0_concat16',
                            'ssd0_concat17',
                            'ssd0_concat18',
                            'ssd0_concat19',
                            'ssd0_concat20',
                            'ssd0_concat21',
                            'ssd0_concat22',
                            'ssd0_concat23']


    if calib_mode == 'none':
        raise ValueError('Not Support')
    else:
        logger.info('Creating GluonDataloader for reading detection dataset')
        val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
        val_data = get_dataloader(
            val_dataset, args.data_shape, args.batch_size, args.num_workers)

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
