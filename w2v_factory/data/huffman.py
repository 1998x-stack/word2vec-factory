from __future__ import annotations
from typing import List, Tuple
import heapq
from dataclasses import dataclass

@dataclass
class HuffmanNode:
    freq: int
    idx: int  # >=0: leaf(word index), <0: internal node id
    left: int = -1
    right: int = -1
    parent: int = -1

def build_huffman_codes(counts: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """构建 Huffman 树并为每个词返回 (path_nodes, path_codes)。
    Returns:
        paths: 每个词到根的节点索引列表（使用内部节点索引，从0开始编号）。
        codes: 对应每条边的 0/1 编码。
    说明：
      - 我们仅为内部节点分配输出向量（与 word2vec 原版一致）。
      - 内部节点数 = 词表大小 - 1。
    """
    vocab_size = len(counts)
    # 初始化叶子节点（词），内部节点 id 从 vocab_size 开始
    pq = []
    nodes: List[HuffmanNode] = []
    for i, c in enumerate(counts):
        heapq.heappush(pq, (c, len(nodes)))
        nodes.append(HuffmanNode(freq=c, idx=i))

    next_internal_id = vocab_size
    while len(pq) > 1:
        (f1, n1_id) = heapq.heappop(pq)
        (f2, n2_id) = heapq.heappop(pq)
        # 创建内部节点
        internal_node = HuffmanNode(freq=f1 + f2, idx=-(next_internal_id + 1), left=n1_id, right=n2_id)
        nodes.append(internal_node)
        parent_id = len(nodes) - 1
        nodes[n1_id].parent = parent_id
        nodes[n2_id].parent = parent_id
        heapq.heappush(pq, (internal_node.freq, parent_id))
        next_internal_id += 1

    # 最后一个是根
    root_id = pq[0][1]
    # 为每个词回溯路径与编码（左=0，右=1）
    paths: List[List[int]] = [[] for _ in range(vocab_size)]
    codes: List[List[int]] = [[] for _ in range(vocab_size)]
    for wid in range(vocab_size):
        path, code = [], []
        nid = wid
        while nodes[nid].parent != -1:
            p = nodes[nid].parent
            if nodes[p].left == nid:
                code.append(0)
            else:
                code.append(1)
            path.append(p)
            nid = p
        paths[wid] = path[::-1]
        codes[wid] = code[::-1]
    return paths, codes
