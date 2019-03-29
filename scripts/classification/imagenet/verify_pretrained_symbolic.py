import argparse, os, math, time

import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.nn import Block, HybridBlock
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='Imagenet directory for validation.')
    parser.add_argument('--rec-dir', type=str, default='',
                        help='recio directory for validation.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--model-prefix', type=str, default='resnet50_v1',
                        help='Prefix of converted model.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='input shape of the image, default is 224.')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='The ratio for crop and input size, for validation dataset only')
    parser.add_argument('--params-file', type=str,
                        help='local parameter file to load, instead of pre-trained weight.')
    parser.add_argument('--dtype', type=str,
                        help='training data type')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()

    CHANNEL_COUNT = 3

    batch_size = opt.batch_size
    classes = 1000

    num_gpus = opt.num_gpus
    if num_gpus > 0:
        batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    input_size = opt.input_size
    pretrained = True if not opt.params_file else False

    kwargs = {'ctx': ctx, 'pretrained': pretrained, 'classes': classes}
    if opt.model_prefix.startswith('resnext'):
        kwargs['use_se'] = opt.use_se

    sym, arg_params, aux_params = mx.model.load_checkpoint(opt.model_prefix, 0)
    net = mx.mod.Module(sym, label_names=[], context=ctx)
    data_shape = [opt.batch_size, CHANNEL_COUNT, opt.input_size, opt.input_size]
    net.bind(data_shapes=[("data", data_shape)], inputs_need_grad=False, for_training=False)
    net.set_params(arg_params=arg_params, aux_params=aux_params)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    """
    Aligning with TF implementation, the default crop-input
    ratio set as 0.875; Set the crop as ceil(input-size/ratio)
    """
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size/crop_ratio))

    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    def test(ctx, val_data, mode='image'):
        acc_top1.reset()
        acc_top5.reset()
        if not opt.rec_dir:
            num_batch = len(val_data)
        num = 0
        start = time.time()
        for i, batch in enumerate(val_data):
            if mode == 'image':
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            else:
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            for x_ in data:
                net.forward(mx.io.DataBatch([x_]))
            outputs = net.get_outputs()
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            if not opt.rec_dir:
                print('%d / %d : %.8f, %.8f'%(i, num_batch, 1-top1, 1-top5))
            else:
                print('%d : %.8f, %.8f'%(i, 1-top1, 1-top5))
            num += batch_size
        end = time.time()
        speed = num / (end - start)
        print('Throughput is %f img/sec.'% speed)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (1-top1, 1-top5)

    if not opt.rec_dir:
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        imgrec = os.path.join(opt.rec_dir, 'val.rec')
        imgidx = os.path.join(opt.rec_dir, 'val.idx')
        val_data = mx.io.ImageRecordIter(
            path_imgrec         = imgrec,
            path_imgidx         = imgidx,
            preprocess_threads  = num_workers,
            batch_size          = batch_size,

            resize              = resize,
            data_shape          = (3, input_size, input_size),
            mean_r              = 123.68,
            mean_g              = 116.779,
            mean_b              = 103.939,
            std_r               = 58.393,
            std_g               = 57.12,
            std_b               = 57.375
        )

    if not opt.rec_dir:
        err_top1_val, err_top5_val = test(ctx, val_data, 'image')
    else:
        err_top1_val, err_top5_val = test(ctx, val_data, 'rec')
    print(err_top1_val, err_top5_val)