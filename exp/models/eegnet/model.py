from .config import CONFIG
from braindecode.models import EEGNet

def build_model() -> EEGNet:
    model = EEGNet(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        sfreq=CONFIG["sfreq"],
        n_outputs=CONFIG["n_classes"],
        final_conv_length="auto",
        final_layer_with_constraint=True,
    )
    
    return model

"""EEGNet model from Lawhern et al (2018) [Lawhern2018]_.

    :bdg-success:`Convolution`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/15/5/056013/revision2/jneaace8cf01_hr.jpg
        :align: center
        :alt: EEGNet Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    EEGNet is a compact convolutional network designed for EEG decoding with a pipeline that mirrors classical EEG processing:
    - (i) learn temporal frequency-selective filters,
    - (ii) learn spatial filters for those frequencies, and
    - (iii) condense features with depthwise-separable convolutions before a lightweight classifier.

    The architecture is deliberately small (temporal convolutional and spatial patterns) [Lawhern2018]_.

    .. rubric:: Macro Components

    - **Temporal convolution**
      Temporal convolution applied per channel; learns ``F1`` kernels that act as data-driven band-pass filters.
    - **Depthwise Spatial Filtering.**
      Depthwise convolution spanning the channel dimension with ``groups = F1``,
      yielding ``D`` spatial filters for each temporal filter (no cross-filter mixing).
    - **Norm-Nonlinearity-Pooling (+ dropout).**
      Batch normalization → ELU → temporal pooling, with dropout.
    - **Depthwise-Separable Convolution Block.**
      (a) depthwise temporal conv to refine temporal structure;
      (b) pointwise 1x1 conv to mix feature maps into ``F2`` combinations.
    - **Classifier Head.**
      Lightweight 1x1 conv or dense layer (often with max-norm constraint).

    .. rubric:: Convolutional Details

    - **Temporal.** The initial temporal convs serve as a *learned filter bank*:
      long 1-D kernels (implemented as 2-D with singleton spatial extent) emphasize oscillatory bands and transients.
      Because this stage is linear prior to BN/ELU, kernels can be analyzed as FIR filters to reveal each feature's spectrum [Lawhern2018]_.

    - **Spatial.** The depthwise spatial conv spans the full channel axis (kernel height = #electrodes; temporal size = 1).
      With ``groups = F1``, each temporal filter learns its own set of ``D`` spatial projections—akin to CSP, learned end-to-end and
      typically regularized with max-norm.

    - **Spectral.** No explicit Fourier/wavelet transform is used. Frequency structure
      is captured implicitly by the temporal filter bank; later depthwise temporal kernels act as short-time integrators/refiners.

    .. rubric:: Additional Comments

    - **Filter-bank structure:** Parallel temporal kernels (``F1``) emulate classical filter banks; pairing them with frequency-specific spatial filters
      yields features mappable to rhythms and topographies.
    - **Depthwise & separable convs:** Parameter-efficient decomposition (depthwise + pointwise) retains power while limiting overfitting
      [Chollet2017]_ and keeps temporal vs. mixing steps interpretable.
    - **Regularization:** Batch norm, dropout, pooling, and optional max-norm on spatial kernels aid stability on small EEG datasets.
    - The v4 means the version 4 at the arxiv paper [Lawhern2018]_.


    Parameters
    ----------
    final_conv_length : int or "auto", default="auto"
        Length of the final convolution layer. If "auto", it is set based on n_times.
    pool_mode : {"mean", "max"}, default="mean"
        Pooling method to use in pooling layers.
    F1 : int, default=8
        Number of temporal filters in the first convolutional layer.
    D : int, default=2
        Depth multiplier for the depthwise convolution.
    F2 : int or None, default=None
        Number of pointwise filters in the separable convolution. Usually set to ``F1 * D``.
    depthwise_kernel_length : int, default=16
        Length of the depthwise convolution kernel in the separable convolution.
    pool1_kernel_size : int, default=4
        Kernel size of the first pooling layer.
    pool2_kernel_size : int, default=8
        Kernel size of the second pooling layer.
    kernel_length : int, default=64
        Length of the temporal convolution kernel.
    conv_spatial_max_norm : float, default=1
        Maximum norm constraint for the spatial (depthwise) convolution.
    activation : nn.Module, default=nn.ELU
        Non-linear activation function to be used in the layers.
    batch_norm_momentum : float, default=0.01
        Momentum for instance normalization in batch norm layers.
    batch_norm_affine : bool, default=True
        If True, batch norm has learnable affine parameters.
    batch_norm_eps : float, default=1e-3
        Epsilon for numeric stability in batch norm layers.
    drop_prob : float, default=0.25
        Dropout probability.
    final_layer_with_constraint : bool, default=False
        If ``False``, uses a convolution-based classification layer. If ``True``,
        apply a flattened linear layer with constraint on the weights norm as the final classification step.
    norm_rate : float, default=0.25
        Max-norm constraint value for the linear layer (used if ``final_layer_conv=False``).

    References
    ----------
    .. [Lawhern2018] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
        Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional
        neural network for EEG-based brain–computer interfaces. Journal of
        neural engineering, 15(5), 056013.
    .. [Chollet2017] Chollet, F., *Xception: Deep Learning with Depthwise Separable
        Convolutions*, CVPR, 2017.

"""




    
