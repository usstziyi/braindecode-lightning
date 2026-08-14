from .config import CONFIG
from braindecode.models import Deep4Net

def build_model() -> Deep4Net:
    model = Deep4Net(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        sfreq=CONFIG["sfreq"],
        n_outputs=CONFIG["n_classes"],
        final_conv_length="auto",
        n_filters_time=CONFIG["n_filters_time"],
        n_filters_spat=CONFIG["n_filters_spat"],
        filter_time_length=CONFIG["filter_time_length"],
        pool_time_length=CONFIG["pool_time_length"],
        pool_time_stride=CONFIG["pool_time_stride"],
        n_filters_2=CONFIG["n_filters_2"],
        filter_length_2=CONFIG["filter_length_2"],
        n_filters_3=CONFIG["n_filters_3"],
        filter_length_3=CONFIG["filter_length_3"],
        n_filters_4=CONFIG["n_filters_4"],
        filter_length_4=CONFIG["filter_length_4"],
        activation_first_conv_nonlin=CONFIG["activation_first_conv_nonlin"],
        first_pool_mode=CONFIG["first_pool_mode"],
        first_pool_nonlin=CONFIG["first_pool_nonlin"],
        activation_later_conv_nonlin=CONFIG["activation_later_conv_nonlin"],
        later_pool_mode=CONFIG["later_pool_mode"],
        later_pool_nonlin=CONFIG["later_pool_nonlin"],
        drop_prob=CONFIG["drop_prob"],
        split_first_layer=CONFIG["split_first_layer"],
        batch_norm=CONFIG["batch_norm"],
        batch_norm_alpha=CONFIG["batch_norm_alpha"],
        stride_before_pool=CONFIG["stride_before_pool"],
    )

    return model

"""Deep4Net model from Schirrmeister et al (2017) [Schirrmeister2017]_.

    :bdg-success:`Convolution`

    .. figure:: https://onlinelibrary.wiley.com/cms/asset/fc200ccc-d8c4-45b4-8577-56ce4d15999a/hbm23730-fig-0001-m.jpg
        :align: center
        :alt: Deep4Net Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    Deep4Net (Deep ConvNet) is a deep convolutional network for EEG decoding
    introduced in [Schirrmeister2017]_, the same paper as ShallowFBCSPNet.
    It consists of four convolutional blocks that learn hierarchical
    temporal and spatial features, followed by a classification head.

    .. rubric:: Macro Components

    - **First block (temporal + spatial conv)**
      A temporal convolution (learned band-pass filters) followed by a
      spatial convolution spanning all channels (learned spatial patterns),
      then batch norm, ELU and max pooling.
    - **Blocks 2-4 (deep temporal conv)**
      Three additional temporal convolutional blocks with increasing
      filter counts (50/100/200), each followed by batch norm, ELU and
      max pooling, extracting progressively higher-level features.
    - **Classifier Head.**
      A final convolution acts as the classifier on the flattened features.

    Parameters
    ----------
    final_conv_length : int or "auto", default="auto"
        Length of the final convolution layer. If "auto", it is set based on n_times.
    n_filters_time : int, default=25
        Number of temporal filters in layer 1.
    n_filters_spat : int, default=25
        Number of spatial filters in layer 1.
    filter_time_length : int, default=10
        Length of the temporal filter in layer 1.
    pool_time_length : int, default=3
        Length of temporal pooling filter.
    pool_time_stride : int, default=3
        Length of stride between temporal pooling filters.
    n_filters_2 : int, default=50
        Number of temporal filters in layer 2.
    filter_length_2 : int, default=10
        Length of the temporal filter in layer 2.
    n_filters_3 : int, default=100
        Number of temporal filters in layer 3.
    filter_length_3 : int, default=10
        Length of the temporal filter in layer 3.
    n_filters_4 : int, default=200
        Number of temporal filters in layer 4.
    filter_length_4 : int, default=10
        Length of the temporal filter in layer 4.
    activation_first_conv_nonlin : nn.Module, default=nn.ELU
        Activation after the first convolution layer.
    first_pool_mode : {"max", "mean"}, default="max"
        Pooling mode in layer 1.
    first_pool_nonlin : nn.Module, default=nn.Identity
        Activation after pooling in layer 1.
    activation_later_conv_nonlin : nn.Module, default=nn.ELU
        Activation after later convolution layers.
    later_pool_mode : {"max", "mean"}, default="max"
        Pooling mode in later layers.
    later_pool_nonlin : nn.Module, default=nn.Identity
        Activation after pooling in later layers.
    drop_prob : float, default=0.5
        Dropout probability.
    split_first_layer : bool, default=True
        Whether to split the first convolution into separate time and spatial convs.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    batch_norm_alpha : float, default=0.1
        Momentum for the running statistics in batch norm.
    stride_before_pool : bool, default=False
        Whether to apply a strided conv before pooling.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
        L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... &
        Ball, T. (2017). Deep learning with convolutional neural networks for EEG
        decoding and visualization. Human brain mapping, 38(11), 5391-5420.

"""



    
