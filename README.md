# Expert Parallelism Load Balancer (EPLB)

EPLB computes expert replica counts and GPU placements for MoE models. It replicates heavily loaded experts and
distributes their replicas across GPUs to balance load. Hierarchical placement keeps each expert group on one node
to reduce cross-node communication under group-limited routing.

The implementation is in [eplb.py](eplb.py). Supply a load estimate for every logical expert and layer, such as a
moving average of token counts. Each expert's estimated load is divided equally among its replicas.

## The Algorithm

The relationship between expert groups and server nodes selects the placement policy.

### Hierarchical Load Balancing

When `groups % nodes == 0`:

1. Assign expert groups to nodes, with equal group counts and balanced estimated loads.
2. Add expert replicas within each node, prioritizing the highest load per replica.
3. Pack replicas onto the node's GPUs, then refine the placement through pair swaps.

Each GPU receives the same number of replicas. This policy suits prefilling with a smaller expert-parallel size.

### Global Load Balancing

When the group count is not divisible by the node count, replicate and pack experts across all GPUs. Pair swaps
also operate across all GPUs. This policy suits decoding with a larger expert-parallel size.

### Improvements

**Heap-based allocation.** A max-heap selects the expert to replicate; a min-heap selects the lightest pack with
free slots. Each allocation updates one heap entry.

| Operation per layer | Original | Improved |
| --- | --- | --- |
| Add `K` replicas to `E` experts | Scan experts each time: `O(KE)` | Heap: `O(E + K log E)` |
| Place `N` objects into `G` packs, excluding sorting | Scan packs each time: `O(NG)` | Heap: `O(N log G)` |

Packing sorts objects by descending load in `O(N log N)` before placement.

**Pair-swap refinement.** After packing, compare the heaviest GPU with lighter GPUs and choose the pair swap with
the largest reduction in squared loads. A binary search finds swap candidates near half the load gap. Refinement
runs for up to 32 swaps per layer and node in hierarchical mode, or per layer across all GPUs in global mode.

For GPU loads `H > L`, an accepted swap transfers a net load `d` with `0 < d < H - L`. The new loads are `H - d`
and `L + d`; the squared-load sum decreases by `2d(H - L - d)`. Each swap preserves GPU replica counts and keeps
the peak load at or below its previous value.

**CPU arithmetic.** Planning uses `float64` and detaches load estimates from autograd. Heap operations use Python
numbers, and the results are converted to tensors after allocation.

## Interface and Example

Install PyTorch and run the example from this directory. The entry point is `eplb.balance`.

| Parameter | Meaning |
| --- | --- |
| `loads` | Nonempty `[layers, experts]` tensor of finite, nonnegative loads |
| `slots` | Total physical replicas per layer; at least the logical expert count and divisible by `gpus` |
| `groups` | Number of equal-sized expert groups; divides the logical expert count |
| `nodes` | Number of server nodes; divides `gpus` |
| `gpus` | Total number of GPUs |

All counts are positive integers. Invalid inputs raise `ValueError`.

The function returns three CPU `int64` tensors:

| Output | Shape | Meaning |
| --- | --- | --- |
| `slot_map` | `[layers, slots]` | Logical expert ID at each physical slot |
| `expert_map` | `[layers, experts, max_copies]` | Physical slots for each expert, padded with `-1` |
| `copies` | `[layers, experts]` | Replica count for each expert |

`max_copies` is the largest value in `copies`. Each block of `slots // gpus` slots in `slot_map`
belongs to one GPU; GPUs are ordered by node.

This example has two layers with 12 logical experts each. Four extra replicas bring each layer to 16 physical
slots, distributed across two nodes with four GPUs per node.

``` python
import torch
import eplb

loads = torch.tensor([
    [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
    [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
])

slot_map, expert_map, copies = eplb.balance(loads, slots=16, groups=4, nodes=2, gpus=8)
print(slot_map)

# Output:
# tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
#         [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])
```

## Measured Results

The comparison used Windows, Python 3.13.9, CPU PyTorch 2.14.1, and one PyTorch thread. Each timing is the median
of three runs. Both configurations have eight layers and 256 logical experts, with random integer loads from 0 to 9999.

| Policy | Replicas / groups / nodes / GPUs | Original planning time | Improved planning time | Speedup |
| --- | --- | --- | --- | --- |
| Hierarchical | 320 / 8 / 8 / 64 | 68.52 ms | 12.63 ms | 5.43x |
| Global | 512 / 8 / 16 / 128 | 3407.82 ms | 98.24 ms | 34.69x |

Across 160 randomized cases covering 309 layers, peak GPU load decreased in 52 layers and stayed equal within
comparison tolerance in 257. The largest reduction was 8.33%. Checks also covered mapping consistency, node
locality, 100 packing cases, 28 invalid inputs, and 40 small replica-allocation cases compared with exhaustive search.

## License

Based on DeepSeek EPLB. The original implementation and this project's modifications use the [MIT License](LICENSE).
