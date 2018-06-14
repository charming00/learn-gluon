import numpy as np
from mxnet import nd
from mxnet import image
import sys

sys.path.append('..')
import utils
from time import time
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import init
import random
import mxnet as mx
import math

classes = ['background', 'p1', 'p2', 'p3', 'p4',
           'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
           'p11', 'p12', 'p13', 'p14', 'p15',
           'p16', 'p17', 'p18', 'p19']
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 64, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0]]
min_coordinate = [[504, 697], [1063, 606], [948, 888], [286, 902], [1079, 1191], [1053, 1507], [1048, 1661],
                  [993, 1729], [1025, 1706], [415, 1391], [1121, 1355], [1138, 1388], [1245, 1294],
                  [1212, 1477], [1170, 1152], [1114, 1756], [635, 1090], [1081, 1100], [350, 1006]]
expand_size = 320
landmark_index = 2
data_root = '../data'
image_root = data_root + '/CephalometricLandmark/Net0Image'
txt_root = data_root + '/CephalometricLandmark/AnnotationsByMD'
model_params_root = data_root + '/CephalometricLandmark/model_params'
rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])
learning_rate = 0.1

for all_round_index in range(1):
    #     image_root = data_root + '/CephalometricLandmark/ContrastImage1'
    #     for expand_size in range(80, 160, 5):
    for landmark_index in range(18, 19):
        #     for landmark_index in [2,3,5,9,12,15]:
        learning_rate = 0.1


        def read_images(dataset_num=0):

            if dataset_num == 0:
                begin_index = 1
                end_index = 151
            elif dataset_num == 1:
                begin_index = 151
                end_index = 301
            else:
                begin_index = 301
                end_index = 401

            data, label = [None] * (end_index - begin_index), [None] * (end_index - begin_index)
            index = 0
            for i in range(begin_index, end_index):
                image_filename = image_root + "/%03d.bmp" % (i)
                txt_filename2 = txt_root + '/400_junior' + "/%03d.txt" % i
                # #         label_image[index] = nd.zeros_like(data[index])

                with open(txt_filename2, 'r') as f:
                    txts1 = f.read().split()
                x = int(txts1[landmark_index].split(',')[0]) - 335
                y = int(txts1[landmark_index].split(',')[1]) - 480
                #         label_image[index][y-expand_size:y+expand_size,x-expand_size:x+expand_size] = colormap[landmark_index+1]
                #             x = int(txts1[landmark_index].split(',')[0])- min_coordinate[landmark_index][0]
                #             y = int(txts1[landmark_index].split(',')[1])- min_coordinate[landmark_index][1]
                minx = x - expand_size
                maxx = x + expand_size
                if minx < 0:
                    minx = 0
                if maxx >= 1600:
                    maxx = 1600 - 1

                miny = y - expand_size
                maxy = y + expand_size
                if miny < 0:
                    miny = 0
                if maxy >= 1920:
                    maxy = 1920 - 1
                data[index] = image.imread(image_filename)
                label[index] = nd.zeros((data[index].shape[0], data[index].shape[1]))
                label[index][miny:maxy, minx:maxx] = 1
                #             label[index] = nd.flip(label[index],0)
                #             print(data[index].shape,label[index].shape)
                index += 1
            return data, label


        def normalize_image(data):
            #             noise = nd.zeros((640, 640, 1))
            #             if random.random() > 0.2:
            #                 mx.random.seed(np.random.randint(1, 1000))
            #                 noise = mx.nd.random.normal(0, 1, shape=(640, 640, 1), dtype=np.float32)
            #             return ((data.astype('float32')+ 10*noise).clip(0, 255) / 255 - rgb_mean) / rgb_std
            return (data.astype('float32') / 255 - rgb_mean) / rgb_std


        class VOCSegDataset(gluon.data.Dataset):

            def __init__(self, dataset_num, crop_size):
                self.crop_size = crop_size
                self.data, self.label = read_images(dataset_num=dataset_num)
                self.data[:] = [normalize_image(im) for im in self.data]

            def __getitem__(self, idx):
                data = self.data[idx]
                label = self.label[idx]

                #                 aug1 = image.HorizontalFlipAug(1)
                #                 aug2 = image.BrightnessJitterAug(1)
                #                 aug3 = image.ContrastJitterAug(1)
                #                 if random.random() > 0.5:
                #                     data = aug1(data)
                #                     label = nd.flip(label,1)
                #                 data = aug2(data)
                return data.transpose((2, 0, 1)), label

            def __len__(self):
                return len(self.data)


        input_shape = (1600, 1920)
        voc_train = VOCSegDataset(0, input_shape)
        voc_test1 = VOCSegDataset(1, input_shape)

        batch_size = 4
        train_data = gluon.data.DataLoader(
            voc_train, batch_size, shuffle=True, last_batch='discard')
        test_data = gluon.data.DataLoader(
            voc_test1, batch_size, last_batch='discard')

        conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
        conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)

        conv.initialize()
        conv_trans.initialize()

        pretrained_net = models.resnet18_v2(pretrained=True)

        net = nn.HybridSequential()
        for layer in pretrained_net.features[:-2]:
            net.add(layer)

        num_classes = len(classes)

        with net.name_scope():
            net.add(
                nn.Conv2D(2, kernel_size=1),
                nn.Conv2DTranspose(2, kernel_size=64, padding=16, strides=32)
            )


        def bilinear_kernel(in_channels, out_channels, kernel_size):
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:kernel_size, :kernel_size]
            filt = (1 - abs(og[0] - center) / factor) * \
                   (1 - abs(og[1] - center) / factor)
            weight = np.zeros(
                (in_channels, out_channels, kernel_size, kernel_size),
                dtype='float32')
            weight[range(in_channels), range(out_channels), :, :] = filt
            return nd.array(weight)


        conv_trans = net[-1]
        conv_trans.initialize(init=init.Zero())
        net[-2].initialize(init=init.Xavier())

        x = nd.zeros((batch_size, 3, *input_shape))
        net(x)

        shape = conv_trans.weight.data().shape
        conv_trans.weight.set_data(bilinear_kernel(*shape[0:3]))

        loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

        ctx = utils.try_all_gpus()
        net.collect_params().reset_ctx(ctx)
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': learning_rate, 'wd': 1e-3})

        utils.train(train_data, test_data, net, loss,
                    trainer, ctx, num_epochs=30)
        model_path = model_params_root + "/net0-mark%02d-resnet18-epochs100.json" % (landmark_index + 1)
        net.save_params(model_path)

