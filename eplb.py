from bisect import bisect_left, insort
from heapq import heapify, heappop, heappush
from typing import Tuple

import torch


def _refine(bins, totals, steps=32):
    """Swap items to reduce squared bin loads at fixed capacity."""
    for items in bins:
        items.sort()

    for _ in range(steps):
        src = max(range(len(bins)), key=totals.__getitem__)
        best_gain = 0.0
        best_swap = None
        for dst in range(len(bins)):
            gap = totals[src] - totals[dst]
            eps = abs(totals[src]) * 1e-12
            if gap <= eps:
                continue
            for src_pos, (src_load, _) in enumerate(bins[src]):
                # The ideal transfer is half the load gap between bins.
                target = src_load - gap / 2
                pos = bisect_left(bins[dst], (target, -1))
                for dst_pos in (pos - 1, pos):
                    if not 0 <= dst_pos < len(bins[dst]):
                        continue
                    delta = src_load - bins[dst][dst_pos][0]
                    if not eps < delta < gap - eps:
                        continue
                    # Half the reduction in the sum of squared bin loads.
                    gain = delta * (gap - delta)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = src, src_pos, dst, dst_pos, delta
        if best_swap is None:
            break
        src, src_pos, dst, dst_pos, delta = best_swap
        src_item = bins[src].pop(src_pos)
        dst_item = bins[dst].pop(dst_pos)
        insort(bins[src], dst_item)
        insort(bins[dst], src_item)
        totals[src] -= delta
        totals[dst] += delta


def _pack(loads: torch.Tensor, packs: int, *, polish: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack [rows, items] loads into equal-sized bins.
    Return bin IDs and positions, each shaped like loads.
    polish enables up to 32 pair swaps per row.
    """
    rows, width = loads.shape
    assert width % packs == 0
    limit = width // packs

    if limit == 1:
        bin_map = torch.arange(
            loads.size(-1), dtype=torch.int64, device=loads.device,
        ).expand(loads.shape)
        positions = torch.zeros_like(loads, dtype=torch.int64)
        return bin_map, positions

    order = loads.sort(-1, descending=True).indices.cpu().tolist()
    values = loads.detach().cpu().tolist()
    all_bins = []
    all_pos = []
    for row in range(rows):
        totals = [0.0] * packs
        bins = [[] for _ in range(packs)]
        queue = [(0.0, bin_id) for bin_id in range(packs)]
        heapify(queue)
        for item in order[row]:
            _, bin_id = heappop(queue)
            bins[bin_id].append((values[row][item], item))
            totals[bin_id] += values[row][item]
            if len(bins[bin_id]) < limit:
                heappush(queue, (totals[bin_id], bin_id))
        if polish and packs > 1:
            _refine(bins, totals)
        bin_map = [-1] * width
        positions = [-1] * width
        for bin_id, items in enumerate(bins):
            for _, item in items:
                bin_map[item] = bin_id
        # Order items by descending input load within each bin.
        offsets = [0] * packs
        for item in order[row]:
            bin_id = bin_map[item]
            positions[item] = offsets[bin_id]
            offsets[bin_id] += 1
        all_bins.append(bin_map)
        all_pos.append(positions)
    return (torch.tensor(all_bins, dtype=torch.int64),
            torch.tensor(all_pos, dtype=torch.int64))


def _replicate(loads: torch.Tensor, slots: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate replicas to experts with the highest load per copy.
    Return slot_map and ranks [layers, slots], plus copies [layers, experts].
    """
    _, experts = loads.shape
    extra = slots - experts
    assert extra >= 0
    device = loads.device
    if extra == 0:
        return (
            torch.arange(experts, dtype=torch.int64, device=device).repeat(
                loads.size(0), 1,
            ),
            torch.zeros_like(loads, dtype=torch.int64),
            torch.ones_like(loads, dtype=torch.int64),
        )
    all_maps, all_ranks, all_counts = [], [], []
    for row in loads.detach().cpu().tolist():
        mapping = list(range(experts))
        ranks = [0] * experts
        counts = [1] * experts
        # Negative loads select the heaviest expert; ties select the lowest ID.
        queue = [(-load, expert) for expert, load in enumerate(row)]
        heapify(queue)
        for _ in range(extra):
            _, expert = heappop(queue)
            mapping.append(expert)
            ranks.append(counts[expert])
            counts[expert] += 1
            heappush(queue, (-row[expert] / counts[expert], expert))
        all_maps.append(mapping)
        all_ranks.append(ranks)
        all_counts.append(counts)
    return (torch.tensor(all_maps, dtype=torch.int64, device=device),
            torch.tensor(all_ranks, dtype=torch.int64, device=device),
            torch.tensor(all_counts, dtype=torch.int64, device=device))


def _layout(loads: torch.Tensor, slots: int, groups: int, nodes: int, gpus: int):
    """Place groups on nodes, then replicate and pack experts onto GPUs."""
    layers, experts = loads.shape
    assert experts % groups == 0
    group_width = experts // groups
    assert groups % nodes == 0
    node_groups = groups // nodes
    assert gpus % nodes == 0
    assert slots % gpus == 0
    gpu_slots = slots // gpus

    def _invert(order: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(order)
        result.scatter_(
            1, order,
            torch.arange(
                order.size(1), dtype=torch.int64, device=order.device,
            ).expand(order.shape),
        )
        return result

    # Step 1: assign expert groups to nodes
    group_loads = loads.unflatten(-1, (groups, group_width)).sum(-1)
    # Use greedy node grouping, then refine GPU placement within each node.
    node_ids, node_pos = _pack(group_loads, nodes, polish=False)
    to_node = (
        ((node_ids * node_groups + node_pos) * group_width).unsqueeze(-1)
        + torch.arange(group_width, dtype=torch.int64, device=node_ids.device)
    ).flatten(-2)
    from_node = _invert(to_node)

    # Step 2: allocate expert replicas within each node
    # [layers * nodes, experts // nodes]
    local_loads = loads.gather(-1, from_node).view(-1, experts // nodes)
    local_map, ranks, local_copies = _replicate(
        local_loads, slots // nodes,
    )

    # Step 3: assign replicas to GPU slots
    # [layers * nodes, slots // nodes]
    slot_loads = (local_loads / local_copies).gather(-1, local_map)
    gpu_ids, gpu_pos = _pack(slot_loads, gpus // nodes)
    to_slot = gpu_ids * gpu_slots + gpu_pos
    from_slot = _invert(to_slot)

    # [layers * nodes, slots // nodes]
    node_map = local_map.gather(-1, from_slot)
    node_map = (
        node_map.view(layers, nodes, -1)
        + torch.arange(
            0, experts, experts // nodes, device=node_ids.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    slot_map = from_node.gather(-1, node_map)
    slot_rank = ranks.gather(-1, from_slot).view(layers, -1)
    copies = local_copies.view(layers, -1).gather(-1, to_node)
    return slot_map, slot_rank, copies


def balance(loads: torch.Tensor, slots: int, groups: int, nodes: int,
            gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Balance expert loads across GPU slots.

    Parameters:
        loads: nonempty [layers, experts] tensor of finite nonnegative loads
        slots: total replicas per layer, at least experts and divisible by gpus
        groups: equal-sized expert group count, dividing experts
        nodes: server node count, dividing gpus
        gpus: total GPU count

    Returns (CPU int64):
        slot_map: [layers, slots], expert ID at each physical slot
        expert_map: [layers, experts, max_copies], slots padded with -1
        copies: [layers, experts], replica count per expert

    max_copies is the largest copy count. Each block of slots // gpus slots
    belongs to one GPU. Invalid loads or topology raise ValueError.
    """
    if not isinstance(loads, torch.Tensor) or loads.ndim != 2:
        raise ValueError("loads must be a two-dimensional torch.Tensor")
    if loads.is_complex() or any(dim == 0 for dim in loads.shape):
        raise ValueError("loads must contain at least one layer and one real-valued expert")
    layers, experts = loads.shape
    for name, value in (
        ("slots", slots), ("groups", groups),
        ("nodes", nodes), ("gpus", gpus),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if slots < experts:
        raise ValueError("slots must be at least the number of logical experts")
    if experts % groups:
        raise ValueError("the number of logical experts must be divisible by groups")
    if gpus % nodes:
        raise ValueError("gpus must be divisible by nodes")
    if slots % gpus:
        raise ValueError("slots must be divisible by gpus")
    loads = loads.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(loads).all().item() or (loads < 0).any().item():
        raise ValueError("loads must contain only finite nonnegative loads")
    if groups % nodes == 0:
        # use hierarchical load-balance policy
        slot_map, slot_rank, copies = _layout(
            loads, slots, groups, nodes, gpus,
        )
    else:
        # use global load-balance policy
        slot_map, slot_rank, copies = _layout(
            loads, slots, 1, 1, gpus,
        )
    max_copies = copies.max().item()
    expert_map: torch.Tensor = torch.full(
        (layers, experts, max_copies),
        -1, dtype=torch.int64, device=copies.device,
    )
    expert_map.view(layers, -1).scatter_(
        -1, slot_map * max_copies + slot_rank,
        torch.arange(
            slots, dtype=torch.int64, device=expert_map.device,
        ).expand(layers, -1),
    )
    return slot_map, expert_map, copies


__all__ = ['balance']
