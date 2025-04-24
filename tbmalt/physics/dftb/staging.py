from __future__ import annotations
from itertools import combinations_with_replacement
from typing import Tuple, Dict, List, Optional
import warnings

import torch
from torch.nn import Parameter

from tbmalt import Geometry
from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.io.skf import Skf
from tbmalt.ml import Feed

from torch import Tensor


class DftbpRepulsiveSpline(Feed):
    """Repulsive spline representation for the repulsive DFTB interaction.

    The repulsive spline implementation matches that used by the DFTB+ package.

    The repulsive potential is partitioned into three regimes and is evaluated
    only within the specified cutoff radius, being zero beyond this distance.

    1. **Short-range exponential head** when distances ≤ first grid point:
       .. math::
           e^{-a_{1} r + a_{2}} + a_{3}

    2. **Intermediate cubic spline body** defined on each interval [r_i, r_{i+1}]:
       .. math::
           c_{0} + c_{1}(r - r_{0}) + c_{2}(r - r_{0})^{2} + c_{3}(r - r_{0})^{3}

    3. **Long-range polynomial tail** between the last spline point and cutoff:
       .. math::
           c_{0} + c_{1}(r - r_{n}) + c_{2}(r - r_{n})^{2}
           + c_{3}(r - r_{n})^{3} + c_{4}(r - r_{n})^{4} + c_{5}(r - r_{n})^{5}

    Arguments:
        grid: Distances for the primary spline segments including the start &
            end of the first & last segments respectively. As such there should
            be n+1 grid points, where n is the number of standard spline
            segments.
        cutoff: Cutoff radius for the spline's tail beyond which interactions
            are assumed to be zero.
        spline_coefficients: An n×4 tensor storing the coefficients for each of
            the primary spline segments.
        exponential_coefficients: A tensor storing the three coefficients of
            the short range exponential region.
        tail_coefficients: A tensor storing the six coefficients of the long -
            range tail region.
    """
    def __init__(
            self, grid: Tensor, cutoff: Tensor, spline_coefficients: Parameter,
            exponential_coefficients: Parameter, tail_coefficients: Parameter):

        super().__init__()
        self.grid = grid
        self.cutoff = cutoff
        self.spline_coefficients = spline_coefficients
        self.exponential_coefficients = exponential_coefficients
        self.tail_coefficients = tail_coefficients

        # Ensure parameters are correctly typed
        if isinstance(grid, Parameter) or grid.requires_grad:
            raise warnings.warn(
                "Setting the grid points as a freely tunable parameter is "
                "strongly advised against as it may result in unexpected "
                "behaviour. Please ensure that the `grid` argument is a "
                "standard `torch.Tensor` type rather than `torch.nn.Parameter` "
                "and that its \"requires_grad\" attribute is set to `False` "
                "unless you are sure of what you are doing.")

        if isinstance(cutoff, Parameter) or grid.requires_grad:
            raise TypeError(
                "The cutoff is not freely tunable parameter. Please ensure "
                "that the `cutoff` argument is a standard `torch.Tensor` type "
                "rather than `torch.nn.Parameter` & that its \"requires_grad\""
                "attribute is set to `False`.")

        if not isinstance(spline_coefficients, Parameter):
            raise TypeError("The spline coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        if not isinstance(exponential_coefficients, Parameter):
            raise TypeError("The exponential coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        if not isinstance(tail_coefficients, Parameter):
            raise TypeError("The tail coefficients must be a "
                            "`torch.nn.Parameter` instance.")

        # Ensure that the tensors are of the correct shape
        if spline_coefficients.ndim != 2 or spline_coefficients.shape[1] != 4:
            raise ValueError("Argument `spline_coefficients` should be an n×4 "
                             "tensor.")

        if spline_coefficients.shape[0] != grid.shape[0] - 1:
            raise ValueError(
                f"{grid.shape[0]} grid values were provided suggesting the "
                f"presence of {grid.shape[0]-1} standard spline segments. "
                f"However, coefficients for {spline_coefficients.shape[0]} "
                f"segments were provided in `spline_coefficients`.")

        if exponential_coefficients.shape != torch.Size([3]):
            raise ValueError(
                f"Expected `exponential_coefficients` argument to be of shape "
                f"torch.Size([3]) but encountered "
                f"{exponential_coefficients.shape}.")

        if tail_coefficients.shape != torch.Size([6]):
            raise ValueError(
                f"Expected `tail_coefficients` argument to be of shape "
                f"torch.Size([6]) but encountered {tail_coefficients.shape}.")

    @property
    def spline_cutoff(self):
        """Cutoff distance of the last primary spline segment."""
        return self.grid[-1]

    @property
    def exponential_cutoff(self):
        """Cutoff distance of the short range exponential region."""
        return self.grid[0]

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)

        # Mask for distances < cutoff
        under_cutoff = distances < self.cutoff

        # Within that subset, distinguish three further conditions:
        # 1) distances > spline_cutoff
        mask_1 = under_cutoff & (distances > self.spline_cutoff)
        # 2) distances > exponential_cutoff and <= spline_cutoff
        mask_2 = under_cutoff & (distances > self.exponential_cutoff
                                 ) & (distances <= self.spline_cutoff)
        # 3) distances <= exp_cutoff
        mask_3 = under_cutoff & (distances <= self.exponential_cutoff)

        # Evaluate the distances in each of the three main distance regimes
        # accordingly.
        results[mask_1] = self._tail(distances[mask_1])
        results[mask_2] = self._spline(distances[mask_2])
        results[mask_3] = self._exponential(distances[mask_3])

        return results

    def _exponential(self, distance: Tensor) -> Tensor:
        """Evaluate the exponential head of the repulsive interaction.

        The short-range exponential part of the repulsive interaction is
        applied when the atoms are closer than the starting distance of the
        first standard spline segment. The repulsive interaction within this
        region is described by the following exponential term:

        .. math::

            e^{-a_{1} r + a_{2}} + a_{3}

        Where "r" (`distances`) is the distance between the atoms and
        :math:`a_{i}` are the exponential coefficients.

        Arguments:
            distance: Distance(s) at which the exponential term is to be
                evaluated.

        Returns:
            repulsive: Exponential repulsive interaction as evaluated at the
                specified distances.
        """
        c = self.exponential_coefficients
        return torch.exp(-c[0] * distance + c[1]) + c[2]

    def _spline(self, distance: Tensor) -> Tensor:
        """Evaluate main spline body of the repulsive interaction.

        Distances between the exponential head and fifth order polynomial
        spline tail are evaluated using a third order polynomial spline of
        the form:

        .. math::

            c_{0}+c_{1}(r-r_{0})+c_{2}(r-r_{0})^{2}+c_{3}(r-r_{0})^{3}

        Where "r" (`distances`) is the distance between the atoms,
        :math:`r_{0}` is the start of the spline segment, and :math:`c_{i}` are
        the spline segment's coefficients.

        Arguments:
            distance: Distance(s) at which the primary spline term is to be
                evaluated.

        Returns:
            repulsive: Primary spline body repulsive interaction as evaluated
                at the specified distances.
        """
        indices = torch.searchsorted(self.grid, distance) - 1
        c = self.spline_coefficients[indices]
        r = distance - self.grid[indices]
        return c[:, 0] + c[:, 1] * r + c[:, 2] * r**2 + c[:, 3] * r**3

    def _tail(self, distance: Tensor) -> Tensor:
        """Evaluate the polynomial tail part of the repulsive interaction.

        Distance between the last standard spline segment's endpoint and the
        cutoff are represented by a tail spline of the form:

        .. math::

            c_{0}+c_{1}(r-r_{0})+c_{2}(r-r_{0})^{2}+c_{3}(r-r_{0})^{3}
                +c_{4}(r-r_{0})^{4}+c_{5}(r-r_{0})^{5}

        Where "r" (`distances`) is the distance between the atoms,
        :math:`r_{0}` is the start of the tail region, and :math:`c_{i}` are
        the tail spline's coefficients.

        Arguments:
            distance: Distance(s) at which the long-range spline tail term is
                to be evaluated.

        Returns:
            repulsive: Spline tail repulsive interaction as evaluated at the
                specified distances.
        """
        c = self.tail_coefficients
        r = distance - self.spline_cutoff
        r_poly = r.unsqueeze(-1).repeat(1, 5).cumprod(-1)
        return c[0] + (c[1:] * r_poly).sum(-1)

    @classmethod
    def from_skf(cls, skf: Skf, requires_grad: bool = False) -> DftbpRepulsiveSpline:
        """Instantiate a `DftbpRepulsiveSpline` instance from a `Skf` object.

        This method will read the repulsive spline data from an `Skf` instance
        representing an sfk formatted file and construct a repulsive spline
        of the form used by the DFTB+ package.

        Arguments:
            skf: An `Skf` instance representing an skf file from which the
                data parameterising the repulsive spline can be read.
            requires_grad: A boolean indicating if the gradient tracking should
                be enabled for the spline's coefficients. [DEFAULT=False]

        Returns:
            repulsive_feed: A `DftbpRepulsiveSpline` instance representing the
                repulsive interaction.

        Notes:
            This assumes the presence of repulsive spline feed in the skf file.
            However, this condition is not guaranteed as some skf files will
            provide a polynomial instead.
        """
        if skf.r_spline is None:
            raise AttributeError(
                f"Skf file {skf} does not define a repulsive spline.")

        return cls.from_r_spline(skf.r_spline, requires_grad=requires_grad)

    @classmethod
    def from_r_spline(
            cls, r_spline: Skf.RSpline, requires_grad: bool = False
            ) -> DftbpRepulsiveSpline:
        """Instantiate a `DftbpRepulsiveSpline` instance from a `Skf.RSpline` object.

        This method will use an `Skf.RSpline` data class to construct a
        repulsive spline of the form used by the DFTB+ package.

        Arguments:
            skf: An `Skf.RSpline` instance parameterising the repulsive spline.
            requires_grad: A boolean indicating if the gradient tracking should
                be enabled for the spline's coefficients. [DEFAULT=False]

        Returns:
            repulsive_feed: A `DftbpRepulsiveSpline` instance representing the
                repulsive interaction.
        """
        return cls(
            r_spline.grid, r_spline.cutoff,
            Parameter(r_spline.spline_coef, requires_grad=requires_grad),
            Parameter(r_spline.exp_coef, requires_grad=requires_grad),
            Parameter(r_spline.tail_coef, requires_grad=requires_grad))


class RepulsiveFeed(Feed):
    # TODO:
    #   - repulsive_feeds should be a parameter dictionary.

    def __init__(self, repulsive_feeds: Dict[Tuple[int, int], DftbpRepulsiveSpline]):
        super().__init__()

        self.repulsive_feeds = repulsive_feeds

    def forward(self, geometry: Geometry) -> Tensor:

        # Tensor to hold the resulting repulsive energy.
        e_rep = torch.zeros(
            geometry.atomic_numbers.shape[0:geometry.atomic_numbers.ndim-1])

        # Iterate over pair-wise interactions by species pair.
        for pair, idx, r in atomic_pair_distances(geometry, True):

            # Identify the repulsive feed associated with the current pair
            feed = self.repulsive_feeds[(pair[0].item(), pair[1].item())]

            # evaluate it at the relevant distances
            e_pairs = feed.forward(r)

            # add the resulting energies to the current total(s)
            e_rep.scatter_add_(0, idx[0], e_pairs)

        return e_rep

    @classmethod
    def from_database(
            cls, path: str, species: List[int],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = False):

        repulsive_feeds = {}

        for pair in combinations_with_replacement(species, r=2):
            skf = Skf.read(path, pair, device=device, dtype=dtype)
            repulsive_feeds[pair] = DftbpRepulsiveSpline.from_skf(skf, requires_grad=requires_grad)

        # noinspection PyTypeChecker
        return cls(repulsive_feeds)