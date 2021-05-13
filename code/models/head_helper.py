"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        fusion="late",
    ):
        """
        ResNetBasicHead takes p pathways as input where p can be greater or
            equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input to the
                ResNetHead.
            num_classes (int): the channel dimension of the output to the
                ResNetHead.
            pool_size (list): list of p the kernel sizes of spatial temporal
                poolings, temporal pool kernel size, height pool kernel size,
                width pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            fusion (string): modality fusion strategy for downstream task.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

        self.fusion = fusion

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            if self.fusion == 'late':
                x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class ResNetPoolingHead(nn.Module):
    """
    ResNe(X)t 3D Pooling head.
    This layer performs a spatioaltemporal pooling. If the inputs are from
    multiple different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        pool_size,
    ):
        """
        ResNetPoolingHead takes p pathways as input where p can be greater or
            equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input to the
                ResNetHead.
            pool_size (list): list of p the kernel sizes of spatial temporal
                poolings, temporal pool kernel size, height pool kernel size,
                width pool kernel size in order.
        """
        super(ResNetPoolingHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.output_size = sum(dim_in)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)

        if not self.training:
            x = x.mean([2, 3, 4])

        x = x.view(x.shape[0], -1)
        return x
