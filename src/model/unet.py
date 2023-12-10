import tensorflow as tf
from tensorflow import keras

class EncoderBlock(keras.layers.Layer):
    def __init__(
        self, filters: int, padding="same", use_maxpool: bool = True,
        name="conv_block", **kwargs
    ):

        super(EncoderBlock, self).__init__(name, **kwargs)

        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu"
        )


        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu"
        )

        self.maxpool = None
        if use_maxpool:
            self.maxpool = keras.layers.MaxPool2D(
                pool_size=(2, 2), strides=2
            )
    
    
    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        if self.maxpool is not None:
            return outputs
        else:
            next_outputs = self.maxpool(outputs)
            return next_outputs, outputs


class DecoderBlock(keras.layers.Layer):
    def __init__(
        self, filters: int, padding="same",
        name="conv_block", **kwargs
    ):

        super(DecoderBlock, self).__init__(name, **kwargs)

        self.upsample = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=(3,3),
            strides=(2,2), padding=padding,
        )

        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu"
        )


        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu"
        )

    
    
    def call(self, last_layer_inputs, skip_connection_inputs):
        outputs = self.upsample(last_layer_inputs)
        outputs = tf.concat([outputs, skip_connection_inputs], axis=-1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(keras.Model):
    def __init__(
        self, filters: int, num_blocks: int, output_activation="sigmoid",
        name="unet", **kwargs):

        super(Unet, self).__init__(name, **kwargs)
    
        self.encoder_blocks = []
        self.decoder_blocks = []

        for _ in range(num_blocks):
            self.encoder_blocks.append(
                EncoderBlock(filters=filters)
            )
            filters *= 2
        
        self.bridge = EncoderBlock(filters=filters, use_maxpool=False)

        for _ in range(num_blocks):
            self.decoder_blocks.append(
                DecoderBlock(filters=filters)
            )
            filters //= 2
        
        self.output_conv = keras.layers.Conv2D(filters=2, kernel_size=(1,1), activation=output_activation)
        
    def call(self, inputs):
        outputs = inputs
        # with tf.init_scope():
        skip_connections = []
        
        for block in self.encoder_blocks:
            outputs, skip_connection = block(outputs)
            skip_connections.append(skip_connection)

        outputs = self.bridge(outputs)

        for skip_connection,block in zip(self.decoder_blocks, skip_connections):
            outputs = block(outputs, skip_connection)
        
        outputs = self.output_conv(outputs)
        return outputs
