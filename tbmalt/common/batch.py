# -*- coding: utf-8 -*-
"""Helper functions for batch operations.

This module contains classes and helper functions associated with batch
construction, handling and maintenance.
"""
from functools import reduce, partial
from typing import Optional, Any, Tuple, List, Union
import numpy as np
from collections import namedtuple
import torch

from tbmalt.common import bool_like
Tensor = torch.Tensor
__sort = namedtuple('sort', ('values', 'indices'))
Sliceable = Union[List[Tensor], Tuple[Tensor]]


def bT(tensor: Tensor) -> Tensor:
    """Dimensionally agnostic "transpose".

    Reverses the dimensions of a tensor like so [m, n, o] -> [o, n, m]. This is
    designed to preserve the original functionality of the `torch.T` operator
    in an effort to maintain dimensional/batch agnosticism. Recent versions of
    PyTorch will only permit the transpose operator to be used on 2D matrices
    which makes dimensionally agnostic treatment of tensors difficult in some
    situations.

    Arguments:
        tensor: the tensor whose dimensions are to be flipped.

    Returns:
        flipped_tensor: the tensor with its dimensions reversed.

    """
    return tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1))


def bT2(tensor: Tensor) -> Tensor:
    """Transposes a tensor and expands it to two dimensions.

    This method performs a transpose on a target tensor via a call to `bT` then
    invokes `torch.atleast_2d` to ensure that the tensor is at least two-
    dimensional. This helps promote batch agnostic programming.

    Note that this is the same as calling `torch.atleast_2d(bT(tensor))` or
    `torch.atleast_2d(tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1)))`.

    Arguments:
        tensor: tensor whose dimensions are to be flipped and expanded.

    Returns:
        modified_tensor: the modified tensor.

    """
    return torch.atleast_2d(bT(tensor))


def pack(tensors: Sliceable, axis: int = 0,
         value: Any = 0, size: Optional[Union[Tuple[int], torch.Size]] = None,
         return_mask: bool = False) -> Union[Tensor, Optional[Tensor]]:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: List of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors should be packed; 0 for first axis -1
            for the last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Size of each dimension to which tensors should be padded. This
            defaults to the largest size encountered along each dimension.
        return_mask: If True, a mask identifying the padding values is
            returned. [DEFAULT=False]

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.
        mask: A tensor that can mask out the padding values. A False value in
            ``mask`` indicates the corresponding entry in ``packed_tensor`` is
            a padding value.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexible than the internal pytorch pack & pad
        functions (at this particular task).

        If ``tensors`` is a `torch.tensor` it will be immedatly returned. This
        helps with batch agnostic programming.

    Examples:
        Multiple tensors can be packed into a single tensor like so:

        >>> from tbmalt.common.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])

        An optional mask identifying the padding values can also be returned:

        >>> packed, mask = pack([torch.tensor([1.]),
        >>>                      torch.tensor([2., 2.]),
        >>>                      torch.tensor([3., 3., 3.])],
        >>>                     return_mask=True)
        >>> print(packed)
        tensor([[1., 0., 0.],
                [2., 2., 0.],
                [3., 3., 3.]])
        >>> print(mask)
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])

    """
    # If "tensors" is already a Tensor then return it immediately as there is
    # nothing more that can be done. This helps with batch agnostic
    # programming.
    if isinstance(tensors, Tensor):
        return tensors

    # Gather some general setup info
    count, device, dtype = len(tensors), tensors[0].device, tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values


    # Tensor to pack into, filled with padding value.
    padded = torch.full((count, *size), value, dtype=dtype, device=device)

    if return_mask:   # Generate the mask if requested.
        mask = torch.full((count, *size), False, dtype=torch.bool,
                          device=device)

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source
        if return_mask:  # Update the mask if required.
            mask[(n, *[slice(0, s) for s in source.shape])] = True

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        ax.insert(axis, 0)  # Re-insert the concatenation axis as specified

        padded = padded.permute(ax)  # Perform the permeation

        if return_mask:  # Perform permeation on the mask is present.
            mask = mask.permute(ax)

    # Return the packed tensor, and the mask if requested.
    return (padded, mask) if return_mask else padded


def pargsort(tensor: Tensor, mask: Optional[bool_like] = None, dim: int = -1
             ) -> Tensor:
    """Returns indices that sort packed tensors while ignoring padding values.

    Returns the indices that sorts the elements of ``tensor`` along ``dim`` in
    ascending order by value while ensuring padding values are shuffled to the
    end of the dimension. This is just a batch capable implementation of the
    `torch.argsort` function.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values restively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: ``indices`` which along the dimension ``dim``.

    Notes:
        This will redirect to `torch.argsort` if no ``mask`` is supplied.

    Examples:

        >>> # Packed array with a padding value of "99"
        >>> array = torch.tensor([
        >>>     [1, 99, 99, 99],
        >>>     [3,  2, 99, 99],
        >>>     [6,  5,  4, 99],
        >>>     [10, 9,  8,  7]
        >>>     ])
        >>>
        >>> # Create a mask that identifies real and padding values
        >>> mask = array != 99
        >>> # Get the sort indices
        >>> sort_indices = pargsort(array, mask)
        >>> print(sort_indices)
        tensor([[0, 1, 2, 3],
                [1, 0, 2, 3],
                [2, 1, 0, 3],
                [3, 2, 1, 0]])
        >>>
        >>> # Use them to sort the array
        >>> array_sorted = array.gather(-1, sort_indices)
        >>> print(array_sorted)
        tensor([[ 1, 99, 99, 99],
                [ 2,  3, 99, 99],
                [ 4,  5,  6, 99],
                [ 7,  8,  9, 10]])
    """
    if mask is None:
        return torch.argsort(tensor, dim=dim)
    else:
        # A secondary sorter is used to reorder the primary sorter so that padding
        # values are moved to the end.
        n = tensor.shape[dim]
        s1 = tensor.argsort(dim)
        s2 = ((torch.arange(n, device=tensor.device)
               + (~mask.gather(dim, s1) * n)).argsort(dim))
        return s1.gather(dim, s2)


def psort(tensor: Tensor, mask: Optional[bool_like] = None, dim: int = -1
          ) -> __sort:
    """Sort a packed ``tensor`` while ignoring any padding values.

    Sorts the elements of ``tensor`` along ``dim`` in ascending order by value
    while ensuring padding values are shuffled to the end of the dimension.
    This is just a batch compatible implimentaiton of the `torch.sort` method.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values respectively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: A namedtuple of (values, indices) is returned, where the values
             are the sorted values and indices are the indices of the elements
             in the original input tensor.

    Notes:
        This will redirect to `torch.sort` if no ``mask`` is supplied.

    Examples:

        >>> # Packed array with a padding value of "99"
        >>> array = torch.tensor([
        >>>     [1, 99, 99, 99],
        >>>     [3,  2, 99, 99],
        >>>     [6,  5,  4, 99],
        >>>     [10, 9,  8,  7]
        >>>     ])
        >>>
        >>> # Create a mask that identifies real and padding values
        >>> mask = array != 99
        >>> # Sort the array
        >>> array_sorted = psort(array, mask)
        >>> print(array_sorted)
        tensor([[ 1, 99, 99, 99],
                [ 2,  3, 99, 99],
                [ 4,  5,  6, 99],
                [ 7,  8,  9, 10]])
    """
    if mask is None:
        return torch.sort(tensor, dim=dim)
    else:
        indices = pargsort(tensor, mask, dim)
        return __sort(tensor.gather(dim, indices), indices)


def merge(tensors: Sliceable, value: Any = 0, axis: int = 0) -> Tensor:
    """Merge two or more packed tensors into a single packed tensor.

    Arguments:
        tensors: Packed tensors that are to be merged.
        value: Value with which the tensor were/are to be padded. [DEFAULT=0]
        axis: Axis along which ``tensors`` are to be stacked. [DEFAULT=0]

    Returns:
        merged: The tensors ``tensors`` merged along the axis ``axis``.

    Warnings:
        Care must be taken to ensure the correct padding value is specified as
        erroneous behaviour may otherwise ensue. As the correct padding value
        cannot be reliably detected in situ it will default to zero.

    Examples:
        >>> # Create a pair of packed tensors
        >>> array_1 = torch.tensor([[1, 99], [3, 2]])
        >>> array_2 = torch.tensor([[6,  5,  4, 99], [10, 9,  8,  7]])
        >>>
        >>> # Merge them into a single array
        >>> merged_array = merge([array_1, array_2], value=99)
        >>> print(merged_array)
        tensor([[ 1, 99, 99, 99],
                [ 3,  2, 99, 99],
                [ 6,  5,  4, 99],
                [10,  9,  8,  7]])
    """

    # Merging is performed along the 0'th axis internally. If a non-zero axis
    # is requested then tensors must be reshaped during input and output.
    if axis != 0:
        tensors = [t.transpose(0, axis) for t in tensors]

    # Tensor to merge into, filled with padding value.
    shapes = torch.tensor([i.shape for i in tensors])
    merged = torch.full(
        (shapes.sum(0)[0], *shapes.max(0).values[1:]),
        value, dtype=tensors[0].dtype, device=tensors[0].device)

    n = 0  # <- batch dimension offset
    for src, size in zip(tensors, shapes):  # Assign values to tensor
        merged[(slice(n, size[0] + n), *[slice(0, s) for s in size[1:]])] = src
        n += size[0]

    # Return the merged tensor, transposing back as required
    return merged if axis == 0 else merged.transpose(0, axis)


def deflate(tensor: Tensor, value: Any = 0, axis: Optional[int] = None
            ) -> Tensor:
    """Shrinks ``tensor`` to remove extraneous trailing padding values.

    Returns a narrowed view of ``tensor`` containing no superfluous trailing
    padding values. For single systems this is equivalent to removing padding.

    All axes are deflated by default, however ``axis`` can be used to forbid
    the deflation of a specific axis. This permits excess padding to be safely
    excised from a batch without inadvertently removing a system from it. This
    is normally the value supplied to the `pack` method for ``axis``.

    Arguments:
        tensor: Tensor to be deflated.
        value: Identity of padding value. [DEFAULT=0]
        axis: Specifies which, if any, axis is exempt from deflation.
            [DEFAULT=None]

    Returns:
        deflated: ``tensor`` after deflation.

    Note:
        Only trailing padding values will be culled; i.e. columns will only be
        removed from the end of a matrix, not the start or the middle.

        Deflation cannot be performed on one dimensional systems when ``axis``
        is not `None`.

    Examples:
        `deflate` can be used to remove unessiary padding from a batch:

        >>> from tbmalt.common.batch import deflate
        >>> over_packed = torch.tensor([
        >>>     [0, 1, 2, 0, 0, 0],
        >>>     [3, 4, 5, 6, 0, 0],
        >>> ])

        >>> print(deflate(over_packed, value=0, axis=0))
        tensor([[0, 1, 2, 0],
                [3, 4, 5, 6]])

        or to remove padding from a system which was once part of a batch:

        >>> packed = torch.tensor([
        >>>     [0, 1, 0, 0],
        >>>     [3, 4, 0, 0],
        >>>     [0, 0, 0, 0],
        >>>     [0, 0, 0, 0]])

        >>> print(deflate(packed, value=0))
        tensor([[0, 1],
                [3, 4]])

    Warnings:
        Under certain circumstances "real" elements may be misidentified as
        padding values if they are equivalent. However, such a complication
        can be mitigated though the selection of an appropriate padding value.

    Raises:
         ValueError: If ``tensor`` is 0 dimensional, or 1 dimensional when
            ``axis`` is not None.
    """

    # Check shape is viable.
    if axis is not None and tensor.ndim <= 1:
        raise ValueError(
            'Tensor must be at least 2D when specifying an ``axis``.')

    mask = tensor == value
    if axis is not None:
        mask = mask.all(axis)

    slices = []
    if (ndim := mask.ndim) > 1:  # When multidimensional `all` is required
        for dim in reversed(torch.combinations(torch.arange(ndim), ndim-1)):
            # Count Nº of trailing padding values. Reduce/partial used here as
            # torch.all cannot operate on multiple dimensions like numpy.
            v, c = reduce(partial(torch.all, keepdims=True), dim, mask
                          ).squeeze().unique_consecutive(return_counts=True)

            # Slicer will be None if there are no trailing padding values.
            slices.append(slice(None, -c[-1] if v[-1] else None))

    else:  # If mask is one dimensional, then no loop is needed
        v, c = mask.unique_consecutive(return_counts=True)
        slices.append(slice(None, -c[-1] if v[-1] else None))

    if axis is not None:
        slices.insert(axis, ...)  # <- dummy index for batch-axis

    return tensor[slices]


def unpack(tensor: Tensor, value: Any = 0, axis: int = 0) -> Tuple[Tensor]:
    """Unpacks packed tensors into their constituents and removes padding.

    This acts as the inverse of the `pack` operation.

    Arguments:
        tensor: Tensor to be unpacked.
        value: Identity of padding value. [DEFAULT=0]
        axis: Axis along which ``tensor`` was packed. [DEFAULT=0]

    Returns:
        tensors: Tuple of constituent tensors.

    Examples:
        
        >>> # Example packed tensor (padded with "99")
        >>> packed_array = torch.tensor([
        >>>     [1, 99, 99, 99],
        >>>     [3,  2, 99, 99],
        >>>     [6,  5,  4, 99],
        >>>     [10, 9,  8,  7]
        >>> ])
        >>> # Unpack the array into its component sub-arrays
        >>> unpacked_arrays = unpack(packed_array, value=99)
        >>> print(unpacked_arrays)
        # (tensor([1]), tensor([3, 2]),
        #  tensor([6, 5, 4]), tensor([10,  9,  8,  7]))

    """
    return tuple(deflate(i, value) for i in tensor.movedim(axis, 0))


def prepeat_interleave(tensor: Tensor, repeats: Tensor, value=0):
    """Batch operable implementation of `torch.repeat_interleave`.

    Repeats elements of a packed tensor.

    Arguments:
        tensor: the tensor whose elements are to be repeated.
        repeats: integers specifying the number of time each element should be
            repeated. This should be the same size as `tensor`.
        value: the padding value used when packing the tensor.

    Examples:

        >>> # A packed array
        >>> array = torch.tensor([[1.1, 0], [2.2, 3.3]])
        >>> # Number of times each element should be repeated
        >>> repeats = torch.tensor([[1, 0], [2, 3]])
        >>> # Perform the repeat
        >>> repeated = prepeat_interleave(array, repeats, 0)
        >>> print(repeated)
        # tensor([[1.1000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [2.2000, 2.2000, 3.3000, 3.3000, 3.3000]])

    """
    if tensor.ndim <= 1:
        return tensor.repeat_interleave(repeats)
    else:
        assert tensor.shape == repeats.shape
        return pack([i.repeat_interleave(j) for i, j in
                     zip(tensor, repeats)], value=value)
