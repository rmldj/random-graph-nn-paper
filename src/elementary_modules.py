import sympy as sym
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """
    Layer that applies a given function on the input
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class IdentityLayer(nn.Module):
    """
    Layer performing identity mapping
    """
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class AbstractNode(nn.Module):
    """
    Abstract class used to create other modules. The AbstractNode module with the blocktype=='simple' consists of a
    weight sum of the inputs, followed by a ReLu activation, convolution and batch_norm.
    """

    def __init__(self, in_channels, out_channels, num_inputs, kernel_size=3, stride=1, restype="C", blocktype="simple"):
        """
        Constructor of the class.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param num_inputs: number of inputs (ingoing edges). Should be >= 1.
        :param kernel_size: The size of the kernel. Default = 3.
        :param stride: The stride size. Default = 1.
        :param restype: The type of the residual connection. Default = 'C'. If set to None, no residual connection will be added to the node.
        :param blocktype: The type of block of operations performed in the node. Default = 'simple'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.stride = stride
        if num_inputs > 1:
            self.weights = nn.Parameter(torch.randn(num_inputs, 1, 1, 1, 1), requires_grad=True)

        self.blocktype = blocktype
        self.__set_block()
        self.restype = restype
        if self.restype is not None:
            self.__set_residual_connection()

    def __set_block(self):
        if self.blocktype == "simple":
            self.block = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                          padding=self.kernel_size // 2, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels))
        elif self.blocktype == "res":
            self.block = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                          padding=self.kernel_size // 2, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size,
                          padding=self.kernel_size // 2, stride=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            raise ValueError("Unknown blocktype {}".format(self.blocktype))

    def __set_residual_connection(self):
        if self.restype == "A":
            channel_pad = (self.out_channels - self.in_channels) // 2
            self.project = LambdaLayer(lambda x:
                                       F.pad(x[:, :, ::self.stride, ::self.stride],
                                             (0, 0, 0, 0, channel_pad, channel_pad), "constant", 0))
        elif self.restype == "B":
            if self.in_channels != self.out_channels or self.stride > 1:
                self.project = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, 1, stride=self.stride, bias=False),
                    nn.BatchNorm2d(self.out_channels)
                )
            else:
                self.project = IdentityLayer()
        elif self.restype == "C":
            self.project = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            raise ValueError("unknown restype: {}".format(self.restype))

    def forward(self, inputs):
        """
        the forward pass of the module.
        :param inputs: Inputs to the model. Should be a list (if self.num_inputs>1)
        or just single tensor (if self.num_inputs == 1)
        :return: returns the tensor being a result of the forward pass of the network
        """
        x = self.aggregated_sum(inputs)
        y = F.relu(x)
        output = self.block(y)
        if self.restype is None:
            return output
        else:
            return output + self.project(x)

    def aggregated_sum(self, inputs):
        if self.num_inputs > 1:
            if type(inputs) == list:
                assert len(inputs) != 0 and len(inputs) == self.num_inputs, \
                    "inputs length cannot be zero and must much num_inputs: {}".format(self.num_inputs)
                shape = list(inputs[0].size())
                inputs = torch.cat(inputs).view([len(inputs)] + shape)
            x = torch.sum(torch.mul(inputs, torch.sigmoid(self.weights)), dim=0)
        else:
            x = inputs
        return x

    def get_block_count(self):
        '''
        Computes the number of parameters.
        :return: the number of parameters (a scalar) used by this node.
        '''
        if self.blocktype == "simple":
            conv_params = self.in_channels * self.kernel_size * self.kernel_size * self.out_channels
            weights = self.num_inputs if self.num_inputs > 1 else 0
            batch_norm = 2 * self.out_channels
            return conv_params + weights + batch_norm
        elif self.blocktype == "res":
            conv_params = self.in_channels * self.kernel_size * self.kernel_size * self.out_channels
            conv_params += self.out_channels * self.kernel_size * self.kernel_size * self.out_channels
            weights = self.num_inputs if self.num_inputs > 1 else 0
            batch_norm = 4 * self.out_channels
            return conv_params + weights + batch_norm
        else:
            raise ValueError("unknown bloktype: {}".format(self.blocktype))

    @staticmethod
    def __get_block_count_sym(C_in, C_out, num_inputs, kernel_size, blocktype):
        if blocktype == "simple":
            sym_conv = C_in * C_out * sym.sympify(kernel_size) ** 2
            sym_weights = sym.sympify(num_inputs) if num_inputs > 1 else sym.sympify(0.0)
            sym_batch_norm = 2 * C_out
            return sym_conv + sym_weights + sym_batch_norm
        elif blocktype == "res":
            sym_conv = C_in * C_out * sym.sympify(kernel_size) ** 2
            sym_conv = sym_conv + (C_out * C_out * sym.sympify(kernel_size) ** 2)
            sym_weights = sym.sympify(num_inputs) if num_inputs > 1 else sym.sympify(0.0)
            sym_batch_norm = 4 * C_out
            return sym_conv + sym_weights + sym_batch_norm
        else:
            raise ValueError("unknown bloktype: {}".format(blocktype))

    def params_count(self):
        """
        function calculating the number of parameters in the network
        :return: the number of trainable parameters in the module
        """
        block_params = self.get_block_count()
        residual_params = 0
        if self.restype == "C" or (self.restype == "B" and (self.in_channels != self.out_channels or self.stride > 1)):
            residual_params += self.in_channels * self.out_channels
            residual_params += 2 * self.out_channels
        return block_params + residual_params

    @staticmethod
    def params_count_sym(C_in, C_out, num_inputs=1, kernel_size=3, restype="C", blocktype="simple", stride=1):
        """
        function returning symbolic equation for the number of parameters in the module.
        :param C_in: symbolic variable for the number of input channels
        :param C_out: symbolic variable for the number of output channels
        :param num_inputs: number of inputs to the layer (default=1)
        :param kernel_size: the size of the kernel (default=3)
        :param restype: the type of the residual connection (default='C')
        :param blocktype: the type of the node operation block (default='simple')
        :param stride: the stride of the convolution (default=1)
        :return: The symbolic equation defining the number of parameters in the module.
        The C_in and C_out should always be functions of the same, one symbolic variable C
        (i.e C_in = g(C) and C_out = f(C))
        """
        sym_block = AbstractNode.__get_block_count_sym(C_in, C_out, num_inputs, kernel_size, blocktype)
        residual_params = sym.sympify(0.0)
        if restype == "C" or (restype == "B" and (C_in != C_out or stride > 1)):
            residual_params = C_in * C_out + residual_params
            residual_params = 2 * C_out + residual_params
        return sym_block + residual_params


class Node(AbstractNode):
    """
    Class representing a single node in the Net.
    Consist of weighted sum, ReLu, convolution layer and batchnorm.
    Number of input channels is equal to number of output channels.
    """

    def __init__(self, channels, num_inputs, kernel_size=3, restype="C", blocktype="simple"):
        """
        Constructor of the module.
        :param channels: number of the input channels.
        :param num_inputs: number of inputs.
        :param kernel_size: the kernel size in convolution layer. Default = 3.
        :param restype: the type of residual connection.
        :param blocktype: the type of the node block operations.
        """
        super().__init__(channels, channels, num_inputs, kernel_size, restype=restype, blocktype=blocktype)

    @staticmethod
    def params_count_sym(C_in, C_out, num_inputs=1, kernel_size=3, restype="C", blocktype="simple", stride=1):
        return AbstractNode.params_count_sym(C_in, C_out, num_inputs, kernel_size, restype, blocktype, stride)


class Reduce(AbstractNode):
    """
    The module performing spatial dimension reduction.
    Consists of ReLu activation, convolution with stride 2 and batchnorm
    """

    def __init__(self, in_channels, out_channels, reduce_ratio, kernel_size=3, restype="C", blocktype="simple"):
        """
        Constructor of the module.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param reduce_ratio: the reduction ratio.
        :param kernel_size: the size of the kernel in convolution. default = 3.
        :param restype: the type of residual connection.
        :param blocktype: the type of the node block operations.
        """
        super().__init__(in_channels, out_channels, num_inputs=1, kernel_size=kernel_size, stride=reduce_ratio,
                         restype=restype, blocktype=blocktype)

    @staticmethod
    def params_count_sym(C_in, C_out, num_inputs=1, kernel_size=3, restype="C", blocktype="simple", stride=2):
        return AbstractNode.params_count_sym(C_in, C_out, num_inputs, kernel_size, restype, blocktype, stride)


class Input(AbstractNode):
    """
    The module for performing initial channels expansion (or reduction).
    """

    def __init__(self, channels, num_inputs=1, kernel_size=3, restype="C", blocktype="simple"):
        """
        the constructor of the input node. The input channels are assumed to be 3.
        :param channels: number of output channels.
        :param num_inputs: number of inputs (ingoing edges), default = 1.
        :param kernel_size: the size of the kernel in convolution, default = 3.
        :param restype: the type of residual connection.
        :param blocktype: the type of the node block operations.
        """
        super().__init__(3, channels, num_inputs, kernel_size, stride=1, restype=restype, blocktype=blocktype)

    @staticmethod
    def params_count_sym(C_in, C_out, num_inputs=1, kernel_size=3, restype="C", blocktype="simple", stride=1):
        return AbstractNode.params_count_sym(C_in, C_out, num_inputs, kernel_size, restype, blocktype, stride)

    def forward(self, inputs):
        x = super().aggregated_sum(inputs)
        if self.restype == "C":
            y = F.relu(x)
        else:
            y = x
        output = self.block(y)
        if self.restype is None:
            return output
        else:
            return output + self.project(x)


class Output(nn.Module):
    """
    The module performing final prediction head operations. Consists of average pooling
    and a dense layer (with no activation).
    """

    def __init__(self, in_channels, num_outputs=10):
        '''
        The constructor of the module.
        :param in_channels: the number of input channels.
        :param num_outputs: the number of prediction outputs. default = 10.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.linear = nn.Linear(self.in_channels, self.num_outputs)

    def forward(self, inputs):
        """
        Performs the forward pass. averages the outputs over all channels and applies
        a linear layer with bias.
        :param inputs: the inputs.
        :return: The result of the last linear layer without activation.
        """
        # assumes N*C*H*W input shape
        out = F.avg_pool2d(inputs, inputs.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def params_count(self):
        """
        Returns the number of parameters.
        :return:
        """
        return self.in_channels * self.num_outputs + self.num_outputs

    @staticmethod
    def params_count_sym(C, num_outputs):
        """
        Calculates the symbolic number of parameters.
        :param C: the symbolic variable for the number of inputs.
        :return: the symbolic equation for the total number of parameters in this module.
        """
        return C * sym.sympify(num_outputs) + sym.sympify(num_outputs)
