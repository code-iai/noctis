import itertools
import operator

import torch
import torch.nn.functional as F

from src.model.utils import BatchedData

from typing_extensions import Optional, Union, Tuple, Sequence


def compute_semantic_similarity(query: torch.Tensor,
                                reference: torch.Tensor,
                                min_threshold: float = 0.0,
                                chunk_size: int = 16,
                                all_at_once: bool = True) -> torch.Tensor:
    """
    :param torch.Tensor query: [B, cF]
    :param torch.Tensor reference: [O, T, cF]
    :param float min_threshold: Similarity scores under this threshold are set to 0.0
    :param int chunk_size:
    :param bool all_at_once:
    :return: [B, O, T]
    :rtype: torch.Tensor
    """
    num_batch = query.shape[0]

    if num_batch > chunk_size:
        similarity = forward_by_chunk_semantic_similarity(query, reference, min_threshold, chunk_size)     # B x O x T
    else:
        query = F.normalize(query, dim=-1)          # chunk_size x cF
        reference = F.normalize(reference, dim=-1)  # O x T x cF

        # determine the cosine similarity score
        if all_at_once:
            similarity = torch.einsum("bf, otf -> bot", query, reference.to(query.device))   # chunk_size x O x T
        else:
            # chunk-wise per object to use less memory
            similarity = [None] * len(reference)
            for index_obj in range(len(reference)):
                obj_ref = reference[index_obj].to(query.device)      # T x cF
                similarity[index_obj] = torch.einsum("bf, tf -> bt",
                                                     query, obj_ref)    # chunk_size x T
            similarity = torch.stack(similarity, dim=0)     # O x chunk_size x T
            similarity = similarity.permute(1, 0, 2)        # chunk_size x O x T

        # all very small scores are set to 0
        similarity[similarity < min_threshold] = 0.0    # chunk_size x O x T

        similarity.clamp_(min=0.0, max=1.0)

    return similarity   # (B|chunk_size) x O x T


def forward_by_chunk_semantic_similarity(query: torch.Tensor,
                                         reference: torch.Tensor,
                                         min_threshold: float = 0.0,
                                         chunk_size: int = 16,
                                         all_at_once: bool = True) -> torch.Tensor:
    """
    :param torch.Tensor query: [B, cF]
    :param torch.Tensor reference: [O, T, cF]
    :param float min_threshold: Similarity scores under this threshold are set to 0.0
    :param int chunk_size:
    :param bool all_at_once:
    :return: [B, O, T]
    :rtype: torch.Tensor
    """
    batch_query = BatchedData(batch_size=chunk_size, data=query)
    del query   # free memory

    scores = [None] * len(batch_query)
    for index_batch in range(len(batch_query)):
        scores[index_batch] = compute_semantic_similarity(query=batch_query[index_batch],
                                                          reference=reference,
                                                          min_threshold=min_threshold,
                                                          chunk_size=chunk_size,
                                                          all_at_once=all_at_once)    # chunk_size x O x T
    scores = torch.cat(scores, dim=0)   # B x O x T

    return scores.data  # B x O x T


def convert_flat_index_to_coordinate(indices: torch.Tensor, shape: Union[int, Sequence[int], torch.Size]) -> torch.Tensor:
    """
    Converts from flat (0 to d1*...*dn range) index to D1x...xDN. coordinate.

    :param torch.Tensor indices: [B1, ..., BM, D1*...*DN]
    :param Union[int, Sequence[int], torch.Size] shape: Sequence (D1, ... DN)
    :return: Tensor [B1, ..., BM, N] with values/indices as D1x...xDN coordinates.
    :rtype: torch.Tensor
    """
    indices = torch.as_tensor(indices)

    # accept real integer values, even with a float type etc.
    if not indices.to(torch.int64).equal(indices) and not indices.is_complex():
        raise TypeError("Expected 'indices' to have real integer values, but got <{}>".format(indices.dtype))
    indices = indices.to(torch.int64)

    if isinstance(shape, int):
        shape = [shape]
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError("Expected all 'shape' values to be a positive integer, but got type <{}>".format(type(dim)))
        if dim < 0:
            raise ValueError("Expected all 'shape' values to be a positive integer, but got <{}>".format(dim))
    shape = list(shape)

    #return torch.stack(torch.unravel_index(index, shape), dim=-1)   # B1 x ... x BM x N

    coefs = list(reversed(list(itertools.accumulate(reversed(shape[1:] + [1]),
                                                    func=operator.mul))))   # N

    # 'floor_divide' to calculate the coordinate indices followed by modulo to handle negative flat indices
    return indices.unsqueeze(-1).floor_divide(
        torch.as_tensor(coefs, device=indices.device, dtype=torch.int64)).remainder(
        torch.as_tensor(shape, device=indices.device, dtype=torch.int64))   # B1 x ... x BM x N


def patch_roundtrip_distance(q2r_patch_indices: torch.Tensor,
                             r2q_patch_indices: torch.Tensor,
                             patch_shape: Union[int, Tuple[int, int]] = 16) -> torch.Tensor:
    """
    :param  torch.Tensor q2r_patch_indices: [B, O, T, hP*wP]
    :param  torch.Tensor r2q_patch_indices: [B, O, T, mP*nP]
    :param Union[int, Tuple[int, int]] patch_shape:
    :return: [B, O, T, hP*wP, 2]
    :rtype: torch.Tensor
    """
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)

    # cycle consistency (query -> reference -> query)
    q2q_patch_indices = torch.gather(r2q_patch_indices, dim=3, index=q2r_patch_indices)         # B x O x T x (hP*wP)
    q2q_patch_indices_2d = convert_flat_index_to_coordinate(q2q_patch_indices, patch_shape)     # B x O x T x (hP*wP) x 2

    # create ground truth indices
    indices_gt = torch.arange(0, patch_shape[0]*patch_shape[1])             # hP*wP
    indices_gt = convert_flat_index_to_coordinate(indices_gt, patch_shape)  # (hP*wP) x 2
    indices_gt = indices_gt.to(q2r_patch_indices.device)

    return torch.linalg.vector_norm((q2q_patch_indices_2d - indices_gt).float(), dim=-1)    # B x O x T x (hP*wP)


def compute_patch_scores(query: torch.Tensor,
                         reference: torch.Tensor,
                         with_visibility: bool = True,
                         min_visibility_threshold: float = 0.0,
                         with_appearance: bool = True,
                         min_appearance_threshold: float = 0.2,
                         cycle_threshold: float = -1,
                         patch_shape: Union[int, Tuple[int, int]] = 16,
                         chunk_size: int = 16,
                         obj_chunk_size: int = 8) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    :param torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param bool with_visibility:
    :param float min_visibility_threshold:
    :param bool with_appearance:
    :param min_appearance_threshold:
    :param float cycle_threshold:
    :param Union[int, Tuple[int, int]] patch_shape:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: A tuple containing the appearance scores [B, O, T] and visibilty [B, O, T].
    :rtype: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
    """
    num_batch = query.shape[0]

    if num_batch > chunk_size:
        visibility_ratios, appearance_scores = forward_by_chunk_patch_scores(
            query=query,
            reference=reference,
            with_visibility=with_visibility,
            min_visibility_threshold=min_visibility_threshold,
            with_appearance=with_appearance,
            min_appearance_threshold=min_appearance_threshold,
            cycle_threshold=cycle_threshold,
            patch_shape=patch_shape,
            chunk_size=chunk_size,
            obj_chunk_size=obj_chunk_size)      # B x O x T, B x O x T
    else:
        query = F.normalize(query, dim=-1)          # chunk_size x (hP*wP) x cF
        reference = F.normalize(reference, dim=-1)  # O x T x (mP*nP) x cF

        obj_indices = torch.arange(reference.shape[0], device=reference.device)     # O
        if obj_chunk_size < 0:
            # all objects at once
            obj_chunk_size = len(obj_indices)

        # chunk-wise per object to use less memory
        obj_indices = torch.split(obj_indices, obj_chunk_size)    # O' x obj_chunk_size

        num_object_chunks = len(obj_indices)
        visibility_ratios = [None] * num_object_chunks
        appearance_scores = [None] * num_object_chunks

        # determine the patch-wise cosine similarity score
        for (i, indices) in enumerate(obj_indices):
            obj_ref = reference[indices].to(query.device)      # obj_chunk_size x T x (mP*nP) x cF
            similarity = torch.einsum("bif, otjf -> botij", query, obj_ref)  # chunk_size x obj_chunk_size x T x (hP*wP) x (mP*nP)

            if with_visibility:
                # calculate ratio of non occluded patches to non-zero reference patches
                r2q_similarity = torch.max(similarity, dim=-2)[0]                   # chunk_size x obj_chunk_size x T x (mP*nP)

                # all smaller scores are set to 0
                r2q_similarity[r2q_similarity < min_visibility_threshold] = 0.0     # chunk_size x obj_chunk_size x T x (mP*nP)

                num_valid_patches = torch.count_nonzero(r2q_similarity, dim=-1)                     # chunk_size x obj_chunk_size x T
                reference_patch_counter = torch.sum(torch.linalg.vector_norm(obj_ref, dim=-1) > 0.01,
                                                    dim=-1).to(num_valid_patches.device) + 1e-6     # obj_chunk_size x T, just counting non-zeros vector sums is not safe
                visibility_ratios[i] = num_valid_patches / reference_patch_counter[None, :]         # chunk_size x obj_chunk_size x T

            if with_appearance:
                # all smaller scores are set to 0
                similarity[similarity < min_appearance_threshold] = 0.0     # chunk_size x obj_chunk_size x T x (hP*wP) x (mP*nP)

                if cycle_threshold >= 0:
                    # find best query to template and reverse matching
                    q2r_similarity, q2r_patch_indices = torch.max(similarity, dim=-1)   # chunk_size x obj_chunk_size x T x (hP*wP), -||-
                    r2q_similarity, r2q_patch_indices = torch.max(similarity, dim=-2)   # chunk_size x obj_chunk_size x T x (mP*nP), -||-

                    # calculate (q->r->q) cycle distance and if the distance is too large, set score to 0
                    q2q_distance = patch_roundtrip_distance(q2r_patch_indices, r2q_patch_indices, patch_shape)  # chunk_size x obj_chunk_size x T x (hP*wP)
                    similarity[q2q_distance > cycle_threshold] = 0.0

                # calculate average template matching score (only count non-zero query patches)
                q2r_similarity = torch.max(similarity, dim=-1)[0]                                           # chunk_size x obj_chunk_size x T x (hP*wP)
                query_patch_counter = torch.sum(torch.linalg.vector_norm(query, dim=-1) >= 0.1,
                                                dim=-1)     # chunk_size, just counting non-zeros vector sums is not safe
                appearance_scores[i] = torch.sum(q2r_similarity, dim=-1) / query_patch_counter[:, None, None]   # chunk_size x obj_chunk_size x T

        if with_visibility:
            visibility_ratios = torch.cat(visibility_ratios, dim=1).clamp(min=0.0, max=1.0)     # chunk_size x O x T
        else:
            visibility_ratios = None

        if with_appearance:
            appearance_scores = torch.cat(appearance_scores, dim=1).clamp(min=0.0, max=1.0)     # chunk_size x O x T
        else:
            appearance_scores = None

    return visibility_ratios, appearance_scores     # (B|chunk_size) x O x T,  # (B|chunk_size) x O x T


def forward_by_chunk_patch_scores(query: torch.Tensor,
                                  reference: torch.Tensor,
                                  with_visibility: bool = True,
                                  min_visibility_threshold: float = 0.0,
                                  with_appearance: bool = True,
                                  min_appearance_threshold: float = 0.2,
                                  cycle_threshold: float = -1,
                                  patch_shape: Union[int, Tuple[int, int]] = 16,
                                  chunk_size: int = 16,
                                  obj_chunk_size: int = 8) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    :param torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param bool with_visibility:
    :param float min_visibility_threshold:
    :param bool with_appearance:
    :param float min_appearance_threshold:
    :param float cycle_threshold:
    :param Union[int, Tuple[int, int]] patch_shape:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: A tuple containing the appearance scores [B, O, T] and visibility [B, O, T].
    :rtype: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
    """
    batch_query = BatchedData(batch_size=chunk_size, data=query)
    del query   # free memory

    visibility_ratios = [None] * len(batch_query)
    appearance_scores = [None] * len(batch_query)
    for index_batch in range(len(batch_query)):
        visibility_ratios[index_batch], appearance_scores[index_batch] = compute_patch_scores(
            query=batch_query[index_batch],
            reference=reference,
            with_visibility=with_visibility,
            min_visibility_threshold=min_visibility_threshold,
            with_appearance=with_appearance,
            min_appearance_threshold=min_appearance_threshold,
            cycle_threshold=cycle_threshold,
            patch_shape=patch_shape,
            chunk_size=chunk_size,
            obj_chunk_size=obj_chunk_size)      # chunk_size x O x T

    if with_visibility:
        visibility_ratios = torch.cat(visibility_ratios, dim=0)   # B x O x T
    else:
        visibility_ratios = None

    if with_appearance:
        appearance_scores = torch.cat(appearance_scores, dim=0)   # B x O x T
    else:
        appearance_scores = None

    return visibility_ratios, appearance_scores   # B x O x T, B x O x T


def compute_appearance_similarity(query: torch.Tensor,
                                  reference: torch.Tensor,
                                  min_threshold: float = 0.0,
                                  cycle_threshold: float = -1,
                                  patch_shape: Union[int, Tuple[int, int]] = 16,
                                  chunk_size: int = 16,
                                  obj_chunk_size: int = 8) -> torch.Tensor:
    """
    :param torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param float min_threshold:
    :param float cycle_threshold:
    :param Union[int, Tuple[int, int]] patch_shape:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: appearance scores [B, O, T]
    :rtype: torch.Tensor
    """
    return compute_patch_scores(query=query,
                                reference=reference,
                                with_visibility=False,
                                min_visibility_threshold=0.0,
                                with_appearance=True,
                                min_appearance_threshold=min_threshold,
                                cycle_threshold=cycle_threshold,
                                patch_shape=patch_shape,
                                chunk_size=chunk_size,
                                obj_chunk_size=obj_chunk_size)[1]       # (B|chunk_size) x O x T


def forward_by_chunk_appearance_similarity(query: torch.Tensor,
                                           reference: torch.Tensor,
                                           min_threshold: float = 0.0,
                                           cycle_threshold: float = 3,
                                           patch_shape: Union[int, Tuple[int, int]] = 16,
                                           chunk_size: int = 16,
                                           obj_chunk_size: int = 8) -> torch.Tensor:
    """
    :param  torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param float min_threshold:
    :param float cycle_threshold:
    :param Union[int, Tuple[int, int]] patch_shape:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: appearance scores [B, O, T]
    :rtype: torch.Tensor
    """
    return forward_by_chunk_patch_scores(query=query,
                                         reference=reference,
                                         with_visibility=False,
                                         min_visibility_threshold=0.0,
                                         with_appearance=True,
                                         min_appearance_threshold=min_threshold,
                                         cycle_threshold=cycle_threshold,
                                         patch_shape=patch_shape,
                                         chunk_size=chunk_size,
                                         obj_chunk_size=obj_chunk_size)[1]      # B x O x T


def compute_visibility_ratio(query: torch.Tensor,
                             reference: torch.Tensor,
                             min_threshold: float = 0.2,
                             chunk_size: int = 16,
                             obj_chunk_size: int = 8) -> torch.Tensor:
    """
    :param torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param float min_threshold:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: visibility [B, O, T]
    :rtype: torch.Tensor
    """
    return compute_patch_scores(query=query,
                                reference=reference,
                                with_visibility=True,
                                min_visibility_threshold=min_threshold,
                                with_appearance=False,
                                min_appearance_threshold=0.0,
                                chunk_size=chunk_size,
                                obj_chunk_size=obj_chunk_size)[0]       # (B|chunk_size) x O x T


def forward_by_chunk_visibility_ratio(query: torch.Tensor,
                                      reference: torch.Tensor,
                                      min_threshold: float = 0.5,
                                      chunk_size: int = 16,
                                      obj_chunk_size: int = 8) -> torch.Tensor:
    """
    :param torch.Tensor query: [B, hP*wP, pF]
    :param torch.Tensor reference: [O, T, mP*nP, pF]
    :param float min_threshold:
    :param int chunk_size:
    :param int obj_chunk_size:
    :return: visibility [B, O, T]
    :rtype: torch.Tensor
    """
    return forward_by_chunk_patch_scores(query=query,
                                         reference=reference,
                                         with_visibility=True,
                                         min_visibility_threshold=min_threshold,
                                         with_appearance=False,
                                         min_appearance_threshold=0.0,
                                         chunk_size=chunk_size,
                                         obj_chunk_size=obj_chunk_size)[0]      # B x O x T
