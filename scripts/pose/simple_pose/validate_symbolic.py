import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from gluoncv.data.transforms.pose import transform_preds, get_final_preds, flip_heatmap
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform, SimplePoseDefaultValTransform
from gluoncv.utils.metrics.coco_keypoints import COCOKeyPointsMetric
from gluoncv.utils.quantization import *

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/coco',
                    help='training and validation pictures to use.')
parser.add_argument('--num-joints', type=int, required=True,
                    help='Number of joints to detect')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--model-prefix', type=str, required=True,
                    help='prefix of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=str, default='256,192',
                    help='size of the input image size. default is 256,192')
parser.add_argument('--flip-test', action='store_true',
                    help='Whether to flip test input to ensemble results.')
parser.add_argument('--mean', type=str, default='0.485,0.456,0.406',
                    help='mean vector for normalization')
parser.add_argument('--std', type=str, default='0.229,0.224,0.225',
                    help='std vector for normalization')
parser.add_argument('--score-threshold', type=float, default=0,
                    help='threshold value for predicted score.')
parser.add_argument('--calibration', type=bool, default=False)
parser.add_argument('--num-calib-batch', type=int, default=5)
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
opt = parser.parse_args()
logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

batch_size = opt.batch_size
num_joints = 17

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

calib_mode = opt.calib_mode
num_calib_batch = opt.num_calib_batch
quantized_dtype = opt.quantized_dtype
enable_calib_quantize = opt.enable_calib_quantize

def get_data_loader(data_dir, batch_size, num_workers, input_size):

    def val_batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        scale = batch[1]
        center = batch[2]
        score = batch[3]
        imgid = batch[4]
        return data, scale, center, score, imgid

    val_dataset = mscoco.keypoints.COCOKeyPoints(data_dir, aspect_ratio=4./3.,
                                                 splits=('person_keypoints_val2017'))

    meanvec = [float(i) for i in opt.mean.split(',')]
    stdvec = [float(i) for i in opt.std.split(',')]
    transform_val = SimplePoseDefaultValTransform(num_joints=val_dataset.num_joints,
                                                  joint_pairs=val_dataset.joint_pairs,
                                                  image_size=input_size,
                                                  mean=meanvec,
                                                  std=stdvec)
    val_data = gluon.data.DataLoader(
        val_dataset.transform(transform_val),
        batch_size=batch_size, shuffle=False, last_batch='keep',
        num_workers=num_workers)

    return val_dataset, val_data, val_batch_fn

input_size = [int(i) for i in opt.input_size.split(',')]
val_dataset, val_data, val_batch_fn = get_data_loader(opt.data_dir, batch_size,
                                                      num_workers, input_size)
val_metric = COCOKeyPointsMetric(val_dataset, 'coco_keypoints',
                                 data_shape=tuple(input_size),
                                 in_vis_thresh=opt.score_threshold)

model_name = opt.model_prefix
sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
net = mx.mod.Module(sym, label_names=[], context=context)
data_shape = [batch_size, 3] + input_size
net.bind(data_shapes=[("data", data_shape)], inputs_need_grad=False, for_training=False)
net.set_params(arg_params=arg_params, aux_params=aux_params)

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

def calibration(val_data, sym, arg_params, aux_params, data_shape, ctx, logger=None):
    if ctx == [mx.cpu()]:
        logger.info('Calibration on Intel CPU')
        sym = sym.get_backend_symbol('MKLDNN')

    logger.info('batch size = %d for calibration' % batch_size)
    logger.info('sampling %d batches for calibration' % num_calib_batch)

    calib_layer = lambda name: name.endswith('_output')
    excluded_sym_names = ['conv3_fwd']

    if calib_mode == 'none':
        raise ValueError('Not Support')
    else:
        logger.info('Creating GluonDataloader for reading detection dataset')

        qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params, 
                                                        data_shape=data_shape,
                                                        ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                        calib_mode=calib_mode, calib_data=val_data,
                                                        num_calib_batch=num_calib_batch,
                                                        calib_layer=calib_layer, quantized_dtype=quantized_dtype,
                                                        logger=logger)
        if calib_mode == 'entropy':
            suffix = '-entropy'
        elif calib_mode == 'naive':
            suffix = '-naive'
        else:
            raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                             % calib_mode)
        sym_name = '%s-quantized-symbol.json' % (model_name + suffix)
    qsym = qsym.get_backend_symbol('MKLDNN_POST_QUANTIZE')
    save_symbol(sym_name, qsym, logger)
    param_name = '%s-quantized-0000.params' % (model_name + suffix)
    save_params(param_name, qarg_params, aux_params, logger)

def validate(val_data, val_dataset, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    val_metric.reset()

    from tqdm import tqdm
    start = time.time()
    for batch in tqdm(val_data):
        data, scale, center, score, imgid = val_batch_fn(batch, ctx)

        for x_ in data:
            net.forward(mx.io.DataBatch([x_]))
        outputs = net.get_outputs()
        if opt.flip_test:
            data_flip = [nd.flip(X, axis=3) for X in data]
            outputs_flip = [net(X) for X in data_flip]
            outputs_flipback = [flip_heatmap(o, val_dataset.joint_pairs, shift=True) for o in outputs_flip]
            outputs = [(o + o_flip)/2 for o, o_flip in zip(outputs, outputs_flipback)]

        if len(outputs) > 1:
            outputs_stack = nd.concat(*[o.as_in_context(mx.cpu()) for o in outputs], dim=0)
        else:
            outputs_stack = outputs[0].as_in_context(mx.cpu())

        preds, maxvals = get_final_preds(outputs_stack, center.asnumpy(), scale.asnumpy())
        val_metric.update(preds, maxvals, score, imgid)
    end = time.time()
    speed = size / (end - start)
    print('Throughput is %f img/sec.'% speed)
    res = val_metric.get()
    return

if __name__ == '__main__':
    if opt.calibration == True:
        calibration(val_data, sym, arg_params, aux_params, data_shape, context, logger=logger)
    else:
        validate(val_data, val_dataset, net, context)
