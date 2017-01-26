from abc import ABC, abstractmethod
import tensorflow as tf


class AbstractNet(ABC):
    """
    Abstract neural network class, includes basic layer definitions.
    """
    @abstractmethod
    def __init__(self, inputs=None, input_shape=None):
        """
        Abstract constructor to take care of handling the input shapes.

        :param inputs: input tensor or placeholder
        :param input_shape: alternative to inputs, shape of inputs to create a placeholder
        """
        assert inputs is not None or input_shape is not None, 'Either inputs or input_shape has to be given.'

        if inputs is not None:
            self._inputs = inputs
        else:
            inputs = tf.placeholder(tf.float32, shape=input_shape, name='inputs')

    @staticmethod
    def add_conv(incoming, ksize=3, n_filters=64, stride=1, biases=False, nonlinearity=None):
        """
        Adds a convolutional layer to the network.

        :param incoming: incoming tensor
        :param ksize: scalar convolution kernel size
        :param n_filters: number of kernels to learn
        :param stride: scalar stride
        :param biases: if True, adds a trainable bias to the result
        :param nonlinearity: non-linearity to apply to the result
        :return: resulting tensor
        """
        with tf.variable_scope('conv'):
            # construct kernel variable
            in_channels = incoming.get_shape()[-1]
            kernel = tf.get_variable('kernel', dtype='float32',
                                     shape=[ksize, ksize, in_channels, n_filters],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())

            # symmetric stride in image coordinates, no stride in batch and channel dimensions
            strides = [1, stride, stride, 1]

            # apply convolution
            outgoing = tf.nn.conv2d(incoming, filter=kernel, strides=strides, padding='SAME')

            # biases?
            if biases:
                biases = tf.get_variable('biases', shape=[n_filters], dtype='float32',
                                         initializer=tf.constant_initializer(0.0))
                outgoing = outgoing + biases

            # non-linearity?
            if nonlinearity:
                outgoing = nonlinearity(outgoing)

        return outgoing

    @staticmethod
    def add_bn(incoming, is_training):
        """
        Adds a batch normalization layer to the network.

        :param incoming: incoming tensor
        :param is_training: boolean 0D-tensor that specifies whether to use accumulated or batch statistics
        :return: batch normalized tensor
        """
        with tf.variable_scope('batchnorm'):
            outgoing = tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, center=True)

        return outgoing

    @staticmethod
    def add_fc(incoming, out_dims):
        """
        Adds a fully-connected layer with biases and without non-linearity to the network.

        :param incoming: 2D tensor of shape [batch, in_dimensions]
        :param out_dims: number of output dimensions
        :return: tensor of shape [batch, out_dim]
        """
        with tf.variable_scope('fc'):
            in_dims = incoming.get_shape()[-1]
            w = tf.get_variable('W', shape=[in_dims, out_dims],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('b', shape=[out_dims],
                                initializer=tf.zeros_initializer)

            outgoing = tf.nn.xw_plus_b(incoming, w, b)

        return outgoing

    @staticmethod
    def add_maxpool(incoming, ksize=2, stride=2):
        """
        Adds a max-pooling layer to the network.

        :param incoming: incoming tensor of shape [N, H, W, C]
        :param ksize: pooling kernel size
        :param stride: pooling stride
        :return: pooled tensor
        """
        return tf.nn.max_pool(incoming, strides=[1, stride, stride, 1], ksize=[1, ksize, ksize, 1], padding='SAME')

    @staticmethod
    def res_block(incoming, is_training, first_stride=1, filters_factor=1, wide_k=1):
        """
        Adds a residual block to the network. Uses two convolutions in the residual branch.
        Depending on the parameters this may include changes in the number of channels as well as the image dimensions.

        :param incoming: incoming tensor
        :param is_training: boolean tensor for batch normalization
        :param first_stride: stride in first convolution, for decreasing the input resolution
        :param filters_factor: factor to compute the number of output filters from the number of input filters
        :param wide_k: additional factor on the number of filters (for wide residual networks)
        :return: sum of identity/projection and residual branch
        """
        # determine output shape
        in_filters = incoming.get_shape().as_list()[-1]
        out_filters = in_filters * filters_factor
        out_filters *= wide_k
        out_filters = int(out_filters)

        # first part: projection/identity branch and first common batch norm
        if out_filters != in_filters or first_stride > 1:

            # common batchnorm
            with tf.variable_scope('common_bn'):
                incoming = AbstractNet.add_bn(incoming, is_training)
                incoming = tf.nn.relu(incoming)

            # first conv in residual branch
            with tf.variable_scope('res_conv1'):
                residual = AbstractNet.add_conv(incoming, n_filters=out_filters, ksize=3, stride=first_stride)

            # projection branch
            with tf.variable_scope('proj_conv'):
                projection = AbstractNet.add_conv(incoming, n_filters=out_filters, ksize=1, stride=first_stride)
        else:
            # first conv in residual branch
            with tf.variable_scope('res_conv1'):
                residual = AbstractNet.add_bn(incoming, is_training)
                residual = tf.nn.relu(residual)
                residual = AbstractNet.add_conv(residual, n_filters=out_filters, ksize=3, stride=1)

            # identity branch
            projection = incoming

        # second conv in residual branch
        with tf.variable_scope('res_conv2'):
            residual = AbstractNet.add_bn(residual, is_training)
            residual = tf.nn.relu(residual)
            residual = AbstractNet.add_conv(residual, n_filters=out_filters, ksize=3, stride=1)

        # merge branches
        net = residual + projection

        return net


class BatchnormNet(AbstractNet):
    """
    Network class that handles batch normalization flags globally.
    """
    def __init__(self, is_training, inputs=None, input_shape=None):
        """
        Constructor with boolean training flag for batch norm
        :param is_training: boolean training flag tensor
        :param inputs: see AbstractNet
        :param input_shape: see AbstractNet
        """
        super(BatchnormNet, self).__init__(inputs, input_shape)

        self._is_training = is_training

    def add_bn(self, incoming):
        return super(BatchnormNet, self).add_bn(incoming, self._is_training)

    def res_block(self, incoming, first_stride=1, filters_factor=1, wide_k=1):
        return super(BatchnormNet, self).res_block(incoming, self._is_training, first_stride=first_stride,
                                                   filters_factor=filters_factor, wide_k=wide_k)