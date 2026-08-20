from .config import CONFIG
from braindecode.models import ShallowFBCSPNet

def build_model() -> ShallowFBCSPNet:
    model = ShallowFBCSPNet(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        sfreq=CONFIG["sfreq"],
        n_outputs=CONFIG["n_classes"],
        n_filters_time=CONFIG["n_filters_time"],
        filter_time_length=CONFIG["filter_time_length"],
        n_filters_spat=CONFIG["n_filters_spat"],
        pool_time_length=CONFIG["pool_time_length"],
        pool_time_stride=CONFIG["pool_time_stride"],
        final_conv_length="auto",
        conv_nonlin=CONFIG["conv_nonlin"],
        pool_mode=CONFIG["pool_mode"],
        activation_pool_nonlin=CONFIG["activation_pool_nonlin"],
        split_first_layer=CONFIG["split_first_layer"],
        batch_norm=CONFIG["batch_norm"],
        batch_norm_alpha=CONFIG["batch_norm_alpha"],
        drop_prob=CONFIG["drop_prob"],
    )

    return model

"""ShallowFBCSPNet model from Schirrmeister et al (2017) [Schirrmeister2017]_.

    :bdg-success:`Convolution`

    .. figure:: https://www.sciencedirect.com/science/article/pii/S1053811917306050/pd/... 
        :align: center
        :alt: ShallowFBCSPNet Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    ShallowFBCSPNet is a shallow convolutional network designed for EEG decoding,
    closely mirroring the classic Filter Bank Common Spatial Patterns (FBCSP)
    pipeline:
    - (i) learn time-domain band-pass filters,
    - (ii) learn spatial filters (linear combination of channels),
    - (iii) square and pool the resulting features, then log-compress and classify.

    The architecture is deliberately shallow (one temporal conv + one spatial conv),
    designed to make the learned spatial filters interpretable as data-driven
    common spatial patterns [Schirrmeister2017]_.

    .. rubric:: Macro Components

    - **Temporal convolution**
      One temporal convolution over the time axis learns time-domain band-pass filters.
      By default the first layer is split into two separate convs (time then space),
      making the spatial filters directly interpretable as spatial patterns.
    - **Spatial Filtering.**
      A convolution spanning the channel dimension learns spatial filters,
      i.e., linear combinations of all channels.
    - **Square + Pool + Log.**
      The temporal/spatial features are squared (power), then pooled over time
      with a mean pool, and finally compressed with a log nonlinearity.
    - **Classifier Head.**
      A final convolution acts as the classifier on the pooled, log-compressed features.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_outputs : int
        Number of classes (output dimension).
    n_times : int
        Number of time samples per input window.
    n_filters_time : int, default=40
        Number of temporal filters.
    filter_time_length : int, default=25
        Length of the temporal convolution kernel.
    n_filters_spat : int, default=40
        Number of spatial filters.
    pool_time_length : int, default=75
        Pooling kernel size along the time axis.
    pool_time_stride : int, default=15
        Pooling stride along the time axis.
    final_conv_length : int or "auto", default="auto"
        Length of the final convolution layer. If "auto", it is set based on n_times.
    conv_nonlin : nn.Module, default=Square
        Non-linear activation for the convolution layers.
    pool_mode : {"mean", "max"}, default="mean"
        Pooling method to use in the pooling layer.
    activation_pool_nonlin : nn.Module, default=SafeLog
        Non-linear activation applied after pooling.
    split_first_layer : bool, default=True
        Whether to split the first convolution into separate time and spatial convs.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    batch_norm_alpha : float, default=0.1
        Momentum for the running statistics in batch norm.
    drop_prob : float, default=0.5
        Dropout probability.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
        L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... &
        Ball, T. (2017). Deep learning with convolutional neural networks for EEG
        decoding and visualization. Human brain mapping, 38(11), 5391-5420.

"""



    
