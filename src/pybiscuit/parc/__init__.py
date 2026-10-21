"""pybiscuit.parc: parcellation network inference (plan rows B7.2, B7.3).

PyTorch is only required by the submodules that run or convert the network
(``unet``, ``checkpoint``, ``infer``); install it with ``pip install pybiscuit[parc]``.
Importing this package alone never imports torch.  ``fusion`` and the legacy-pickle audit
in ``checkpoint`` do not need it either.
"""
