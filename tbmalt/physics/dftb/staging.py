from __future__ import annotations
from itertools import combinations_with_replacement

from typing import Tuple, Dict, List
import torch

from tbmalt import Geometry
from tbmalt.structures.geometry import atomic_pair_indices
from tbmalt.io.skf import Skf
from tbmalt.ml import Feed

from torch import Tensor


class RepulsiveSpline(Feed):
    def __init__(self, grid: Tensor, cutoff: Tensor, spline_coefficients: Tensor, exponential_coefficients: Tensor,
                 tail_coefficients: Tensor):

        # TODO:
        #   - Permit explicit analytical gradient methods to be added
        #   - Ensure that the coefficients are provided as `Parameter` types
        #       rather than `Tensor` instances.

        super().__init__()
        self.grid = grid
        self.cutoff = cutoff
        self.spline_coefficients = spline_coefficients
        self.exponential_coefficients = exponential_coefficients
        self.tail_coefficients = tail_coefficients

    @property
    def spline_cutoff(self):
        return self.grid[-1]

    @property
    def exponential_cutoff(self):
        return self.grid[0]

    def forward(self, distances: Tensor) -> Tensor:

        results = torch.zeros_like(distances)

        # Mask for distances < cutoff
        under_cutoff = distances < self.cutoff

        # Within that subset, we distinguish three further conditions:
        # 1) distances > spline_cutoff
        mask_1 = under_cutoff & (distances > self.spline_cutoff)
        # 2) distances > exponential_cutoff and <= spline_cutoff
        mask_2 = under_cutoff & (distances > self.exponential_cutoff
                                 ) & (distances <= self.spline_cutoff)
        # 3) distances <= exp_cutoff
        mask_3 = under_cutoff & (distances <= self.exponential_cutoff)

        results[mask_1] = self._tail(distances[mask_1])
        results[mask_2] = self._spline(distances[mask_2])
        results[mask_3] = self._exponential(distances[mask_3])

        return results

    def _spline(self, distance: Tensor) -> Tensor:
        # TODO:
        #   - Add in dimension and type coercion for distance
        #   - Vectorise.
        indices = torch.searchsorted(self.grid, distance) - 1
        c = self.spline_coefficients[indices]
        r = distance - self.grid[indices]
        return c[:, 0] + c[:, 1] * r + c[:, 2] * r**2 + c[:, 3] * r**3

    def _exponential(self, distance: Tensor) -> Tensor:
        c = self.exponential_coefficients
        return torch.exp(-c[0] * distance + c[1]) + c[2]

    def _tail(self, distance: Tensor) -> Tensor:
        # TODO:
        #   - Add in dimension and type coercion for distance
        c = self.tail_coefficients
        r = distance - self.spline_cutoff
        r_poly = r.unsqueeze(-1).repeat(1, 5).cumprod(-1)
        return c[0] + (c[1:] * r_poly).sum(-1)

    @classmethod
    def from_skf(cls, skf: Skf) -> RepulsiveSpline:
        if skf.r_spline is None:
            raise AttributeError(
                f"Skf file {skf} does not define a repulsive spline.")

        return cls.from_r_spline(skf.r_spline)

    @classmethod
    def from_r_spline(cls, r_spline: Skf.RSpline) -> RepulsiveSpline:
        return cls(r_spline.grid, r_spline.cutoff, r_spline.spline_coef,
                   r_spline.exp_coef, r_spline.tail_coef)


class RepulsiveFeed(Feed):
    # TODO:
    #   - repulsive_feeds should be a parameter dictionary.

    def __init__(self, repulsive_feeds: Dict[Tuple[int, int], RepulsiveSpline]):
        super().__init__()

        self.repulsive_feeds = repulsive_feeds

    def forward(self, geometry: Geometry) -> Tensor:

        # Tensor to hold the resulting repulsive energy.
        e_rep = torch.zeros(
            geometry.atomic_numbers.shape[0:geometry.atomic_numbers.ndim-1])

        # Compute distances between all atom pairs and enforce the presence of
        # a batch dimension.
        r = geometry.periodicity.periodic_distances if geometry.is_periodic \
            else geometry.distances

        r = r.view(-1, *r.shape[-3 if geometry.is_periodic else -2:])

        # Iterate over pair-wise interactions by species pair.
        for pair, idx in atomic_pair_indices(geometry, True, True):

            # Identify the repulsive feed associated with the current pair
            feed = self.repulsive_feeds[(pair[0].item(), pair[1].item())]

            # evaluate it at the relevant distances
            e_pairs = feed.forward(r[*idx])

            # add the resulting energies to the current total(s)
            e_rep.scatter_add_(0, idx[0], e_pairs)

        return e_rep

    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs):
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)

        repulsive_feeds = {}

        for pair in combinations_with_replacement(species, r=2):
            skf = Skf.read(path, pair, device=device, dtype=dtype)
            repulsive_feeds[pair] = RepulsiveSpline.from_skf(skf)

        # noinspection PyTypeChecker
        return cls(repulsive_feeds)
