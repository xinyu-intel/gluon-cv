"""Classification Demo script."""
import os
import argparse
import time
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.imagenet import transform_eval
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Test with SSD networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
            "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
        image_list = ['street.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained)

    # export to json and load back with SymbolBlock
    print('Export to JSON...')
    gcv.utils.export_block(args.network, net, preprocess=False, layout='CHW')
    print('Load back from JSON with SymbolBlock')
    net_import = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(args.network),
        ['data'], '{}-0000.params'.format(args.network), fusion=True)

    net_import.collect_params().reset_ctx(ctx = ctx)

    for image in image_list:
        img = mx.image.imread(image)
        img = transform_eval(img)
        start = time.time()
        for i in range(5000):
            pred = net_import(img)
        end = time.time()
        speed = (end - start) / 5000
        print('latency is %f ms.'% (1000 * speed))
        # topK = 5
        # ind = mx.nd.topk(pred, k=topK)[0].astype('int')
        # print('The input picture is classified to be')
        # for i in range(topK):
        #     print('\t[%s], with probability %.3f.'%
        #         (net.classes[ind[i].asscalar()], mx.nd.softmax(pred)[0][ind[i]].asscalar()))
