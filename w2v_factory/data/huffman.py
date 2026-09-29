from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class HuffmanNode:
    freq: int
    left: int = -1
    right: int = -1
    parent: int = -1


def build_huffman_codes(counts: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """构建 Huffman 树并为每个词返回 (path_nodes, path_codes)。

    Returns:
        paths: 每个词从根到叶路径上的内部节点索引，紧凑编号为 [0, V-2]。
        codes: 对应每条边的 0/1 编码。

    说明：
      - 仅内部节点需要输出向量（与原版 word2vec 一致）。
      - V 个词对应 V-1 个内部节点，因此 HS 输出表只需 (V-1) x dim。
    """
    vocab_size = len(counts)
    if any(c <= 0 for c in counts):
        raise ValueError("Huffman counts must be positive")

    pq: list[tuple[int, int]] = []
    nodes: list[HuffmanNode] = []
    for c in counts:
        heapq.heappush(pq, (c, len(nodes)))
        nodes.append(HuffmanNode(freq=c))

    while len(pq) > 1:
        f1, n1_id = heapq.heappop(pq)
        f2, n2_id = heapq.heappop(pq)
        nodes.append(HuffmanNode(freq=f1 + f2, left=n1_id, right=n2_id))
        parent_id = len(nodes) - 1
        nodes[n1_id].parent = parent_id
        nodes[n2_id].parent = parent_id
        heapq.heappush(pq, (f1 + f2, parent_id))

    paths: list[list[int]] = [[] for _ in range(vocab_size)]
    codes: list[list[int]] = [[] for _ in range(vocab_size)]
    for wid in range(vocab_size):
        path: list[int] = []
        code: list[int] = []
        nid = wid
        while nodes[nid].parent != -1:
            parent = nodes[nid].parent
            code.append(0 if nodes[parent].left == nid else 1)
            path.append(parent - vocab_size)
            nid = parent
        paths[wid] = path[::-1]
        codes[wid] = code[::-1]

    return paths, codes
