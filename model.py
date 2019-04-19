from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Reshape, Concatenate

import tensorflow as tf


def make_model():
    x = Input(shape=(1024, 512, 3), name='input')

    conv_1 = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_1')(x)
    bn_1 = BatchNormalization(name='bn_1')(conv_1)
    pool_1 = MaxPool2D(pool_size=(2, 2), name='pool_1')(bn_1)

    conv_2 = Conv2D(filters=48, kernel_size=3, strides=1, padding='same', activation='relu', name='conv_2')(pool_1)
    bn_2 = BatchNormalization(name='bn_2')(conv_2)
    pool_2 = MaxPool2D(pool_size=(2, 2), name='pool_2')(bn_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_3')(pool_2)
    bn_3 = BatchNormalization(name='bn_3')(conv_3)
    conv_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', activation='relu', name='conv_4')(bn_3)
    bn_4 = BatchNormalization(name='bn_4')(conv_4)
    conv_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_5')(bn_4)
    bn_5 = BatchNormalization(name='bn_5')(conv_5)
    pool_3 = MaxPool2D(pool_size=(2, 2), name='pool_3')(bn_5)

    conv_6 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_6')(pool_3)
    bn_6 = BatchNormalization(name='bn_6')(conv_6)
    conv_7 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_7')(bn_6)
    bn_7 = BatchNormalization(name='bn_7')(conv_7)
    conv_8 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_8')(bn_7)
    bn_8 = BatchNormalization(name='bn_8')(conv_8)
    pool_4 = MaxPool2D(pool_size=(2, 2), name='pool_4')(bn_8)

    conv_9 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_9')(pool_4)
    bn_9 = BatchNormalization(name='bn_9')(conv_9)

    route_1 = bn_9  # 12 layer
    reorg_result = Reshape(target_shape=(32, 16, 1024), name='reorg_result')(route_1)

    conv_10 = Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same', activation='relu', name='conv_10')(bn_9)
    bn_10 = BatchNormalization(name='bn_10')(conv_10)
    conv_11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_11')(bn_10)
    bn_11 = BatchNormalization(name='bn_11')(conv_11)
    pool_5 = MaxPool2D(pool_size=(2, 2), name='pool_5')(bn_11)

    conv_12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_12')(pool_5)
    bn_12 = BatchNormalization(name='bn_12')(conv_12)
    conv_13 = Conv2D(filters=512, kernel_size=(1, 1), strides=1, padding='same', activation='relu', name='conv_13')(bn_12)
    bn_13 = BatchNormalization(name='bn_13')(conv_13)
    conv_14 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_14')(bn_13)
    bn_14 = BatchNormalization(name='bn_14')(conv_14)
    conv_15 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_15')(bn_14)
    bn_15 = BatchNormalization(name='bn_15')(conv_15)
    conv_16 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_16')(bn_15)
    bn_16 = BatchNormalization(name='bn_16')(conv_16)

    route_2 = Concatenate(name='concat_1')([reorg_result, bn_16])

    conv_17 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', activation='relu', name='conv_17')(route_2)
    bn_17 = BatchNormalization(name='bn_17')(conv_17)
    conv_18 = Conv2D(filters=75, kernel_size=(1, 1), strides=1, padding='same', name='output')(bn_17)

    return Model(inputs=x, outputs=[conv_18])


if __name__ == '__main__':
    model = make_model()
    model.summary()
