import tensorflow as tf
from tensorflow import keras

class EncoderBlock(keras.layers.Layer):
    def __init__(
        self, filters: int, dropout_rate, use_batchnorm: bool, padding="same", use_maxpool: bool = True,
        name="encoder_block", **kwargs
    ):

        super(EncoderBlock, self).__init__(name=name, **kwargs)

        # self.config = {
        #     "filters": filters,
        #     "padding": padding,
        #     "use_maxpool": use_maxpool
        # }

        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu", use_bias=not use_batchnorm
        )

        self.batchnorm1 = None
        if use_batchnorm:
            self.batchnorm1 = keras.layers.BatchNormalization()
        
        self.dropout = None
        if dropout_rate > 0.0:
            self.dropout = keras.layers.SpatialDropout2D(rate=dropout_rate)

        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu", use_bias=not use_batchnorm
        )

        self.batchnorm2 = None
        if use_batchnorm:
            self.batchnorm2 = keras.layers.BatchNormalization()

        self.maxpool = None
        if use_maxpool:
            self.maxpool = keras.layers.MaxPool2D(
                pool_size=(2, 2), strides=2
            )
        
    
    
    def call(self, inputs):
        outputs = self.conv1(inputs)
        
        if self.batchnorm1 is not None:
            outputs = self.batchnorm1(outputs)
        
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        
        if self.batchnorm2 is not None:
            outputs = self.batchnorm2(outputs)
        outputs = self.conv2(outputs)
        

        if self.maxpool is None:
            return outputs
        else:
            next_outputs = self.maxpool(outputs)
            return next_outputs, outputs
    
    # def get_config(self):
    #     base_config = super().get_config()
    #     return {**base_config, **self.config}


class DecoderBlock(keras.layers.Layer):
    def __init__(
        self, filters: int, dropout_rate: float, use_batchnorm: bool, padding="same",
        name="decoder_block", **kwargs
    ):

        super(DecoderBlock, self).__init__(name=name, **kwargs)

        self.config = {
            "filters": filters,
            "padding": padding
        }


        self.upsample = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=(3,3),
            strides=(2,2), padding=padding, use_bias=not use_batchnorm,
        )

        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu", use_bias=not use_batchnorm,
        )

        self.batchnorm1 = None
        if use_batchnorm:
            self.batchnorm1 = keras.layers.BatchNormalization()
        

        self.dropout = None
        if dropout_rate > 0.0:
            self.dropout = keras.layers.SpatialDropout2D(rate=dropout_rate)

        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3,3),
            padding=padding, activation="relu", use_bias=not use_batchnorm,
        )
        
        self.batchnorm2 = None
        if use_batchnorm:
            self.batchnorm2 = keras.layers.BatchNormalization()
    
    
    def call(self, last_layer_inputs, skip_connection_inputs):
        outputs = self.upsample(last_layer_inputs)
        outputs = tf.concat([outputs, skip_connection_inputs], axis=-1)
        outputs = self.conv1(outputs)
        
        if self.batchnorm1 is not None:
            outputs = self.batchnorm1(outputs)
        
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        
        if self.batchnorm2 is not None:
            outputs = self.batchnorm2(outputs)
        return outputs
    
    


class Unet(keras.Model):
    def __init__(
        self, filters: int, num_blocks: int, dropout_rate: float, use_batchnorm: bool, output_activation="sigmoid",
        name="unet", **kwargs):

        super(Unet, self).__init__(name=name, **kwargs)
    
        self.encoder_blocks = []
        self.decoder_blocks = []

        for _ in range(num_blocks):
            self.encoder_blocks.append(
                EncoderBlock(filters=filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
            )
            filters *= 2
        
        self.bridge = EncoderBlock(filters=filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm, use_maxpool=False)

        for _ in range(num_blocks):
            self.decoder_blocks.append(
                DecoderBlock(filters=filters, dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
            )
            filters //= 2
        
        self.output_conv = keras.layers.Conv2D(filters=1, kernel_size=(1,1), activation=output_activation)
        
    def call(self, inputs):
        outputs = inputs
        # with tf.init_scope():
        skip_connections = []
        
        for block in self.encoder_blocks:
            outputs, skip_connection = block(outputs)
            skip_connections.append(skip_connection)

        outputs = self.bridge(outputs)

        for block, skip_connection in zip(self.decoder_blocks, skip_connections[::-1]):
            outputs = block(outputs, skip_connection)
        
        outputs = self.output_conv(outputs)
        return outputs
