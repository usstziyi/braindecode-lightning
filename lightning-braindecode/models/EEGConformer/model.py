from .config import CONFIG
from braindecode.models import EEGConformer

def build_model() -> EEGConformer:
    model = EEGConformer(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        sfreq=CONFIG["sfreq"],
        n_outputs=CONFIG["n_classes"],
        n_filters_time=CONFIG["n_filters_time"],
        filter_time_length=CONFIG["filter_time_length"],
        pool_time_length=CONFIG["pool_time_length"],
        pool_time_stride=CONFIG["pool_time_stride"],
        drop_prob=CONFIG["drop_prob"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        att_drop_prob=CONFIG["att_drop_prob"],
        final_fc_length="auto",
        return_features=CONFIG["return_features"],
        activation=CONFIG["activation"],
        activation_transfor=CONFIG["activation_transfor"],
    )

    return model

"""EEG-Conformer model from Song et al (2022) [Song2022]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/7333/9053006/9955847/song1-3241970-large.gif
        :align: center
        :alt: EEG-Conformer Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    EEG-Conformer is a *convolution-first* architecture augmented with a
    lightweight Transformer encoder. It first extracts local, low-level
    temporal and spatial features with a compact CNN (similar to the first
    block of ShallowConvNet), then feeds the resulting sequence of
    features through a Transformer encoder to model long-range temporal
    dependencies [Song2022]_.

    .. rubric:: Macro Components

    - **Convolutional stage**
      A temporal convolution (learned band-pass filters) followed by a
      spatial convolution spanning all channels (learned spatial patterns),
      then pooling to produce a compact feature map.
    - **Patch Embedding**
      The feature map is split into patches, each flattened and projected
      to the Transformer embedding dimension.
    - **Transformer Encoder**
      A stack of ``num_layers`` Transformer encoder blocks (multi-head
      self-attention + MLP with GELU), capturing global temporal context.
    - **Classifier Head.**
      The class token / pooled features are fed to a linear classifier
      producing class logits (or raw features if ``return_features=True``).

    Parameters
    ----------
    n_outputs : int
        Number of classes.
    n_chans : int
        Number of EEG channels.
    n_filters_time : int, default=40
        Number of temporal filters in the first convolution layer.
    filter_time_length : int, default=25
        Length of the temporal convolution kernel.
    pool_time_length : int, default=75
        Pooling kernel size along the time axis.
    pool_time_stride : int, default=15
        Pooling stride along the time axis.
    drop_prob : float, default=0.5
        Dropout probability.
    num_layers : int, default=6
        Number of Transformer encoder layers.
    num_heads : int, default=10
        Number of attention heads.
    att_drop_prob : float, default=0.5
        Attention dropout probability.
    final_fc_length : int or "auto", default="auto"
        Length of the final fully-connected layer. If "auto", computed from n_times.
    return_features : bool, default=False
        If True, return the feature vectors instead of class logits
        (useful as a feature encoder, e.g. for EEG2Text).
    activation : nn.Module, default=nn.ELU
        Activation for the convolutional stage.
    activation_transfor : nn.Module, default=nn.GELU
        Activation inside the Transformer encoder (MLP).

    References
    ----------
    .. [Song2022] Song, Y., Zheng, Q., Liu, B., & Gao, X. (2022). EEG Conformer:
        Convolutional Transformer for EEG Decoding and Visualization. IEEE
        Transactions on Neural Systems and Rehabilitation Engineering, 31,
        762-773.

"""



    
