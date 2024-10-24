# -*- coding: utf-8 -*-
"""This module houses the majority of TBMaLT's machine learning code.

The primary machine learning components of TBMaLT are located within this
module.
"""
from abc import ABC
import torch


class Feed(torch.nn.Module, ABC):
    """Abstract base class for all feed components in the TBMaLT framework.

    The `Feed` class serves as a foundational building block for creating
    diverse and flexible components that process inputs to generate outputs.
    These components, referred to as "feeds," encapsulate optimisable
    parameters that can be tuned to enhance the output. This class is designed
    to integrate seamlessly with the PyTorch ecosystem, allowing for the
    utilization of PyTorch's optimizers, device management, and other
    functionalities.

    Subclasses of `Feed` must implement specific methods to define how inputs
    are processed and how the parameters are used. The parameters within each
    `Feed` object can vary significantly, as the class is intended to support a
    wide range of computational processes, from spline interpolations to
    quantum mechanical solvers.

    Key Features:
    - Automatic tracking of optimisable parameters, facilitating integration
      with PyTorch's optimization routines.
    - Support for nested `Feed` objects, allowing for the construction of
      complex computational workflows.
    - Flexibility in input and output definitions, with no enforced method
      signature, enabling subclasses to define methods with varying
      arguments according to their specific needs.

    This class is crucial for the modular design of the framework, enabling
    components to be easily combined, extended, and reused across different
    applications.

    Note:
        Subclasses must implement the `forward` method to define the computation
        performed by the feed. Other methods may be required depending on the
        specific subclass implementation.

    TODO:
     - Documentation.
        - Make sure to remark that it may be necessary to override the `parameters`
          method of the base `torch.nn.Module` class. This is because the
          `parameters` method is used to fetch all optimisation targets.
     - All feeds should be modified to make use of the `.forward` method rather
       than the `__call__` method.

    """
    def __init__(self):
        super().__init__()
