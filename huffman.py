from collections import Counter
import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _Node:
    freq: int
    order: int
    symbol: Any = field(compare=False, default=None)
    left: "_Node | None" = field(compare=False, default=None)
    right: "_Node | None" = field(compare=False, default=None)
    is_leaf: bool = field(compare=False, default=False)


def _build_tree(freq: dict[Any, int]) -> _Node | None:
    if not freq:
        return None

    heap: list[tuple[int, int, _Node]] = []
    for order, (symbol, count) in enumerate(sorted(freq.items(), key=lambda item: item[0])):
        heapq.heappush(heap, (count, order, _Node(count, order, symbol=symbol, is_leaf=True)))

    if len(heap) == 1:
        return heap[0][2]

    next_order = len(heap)
    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        parent = _Node(f1 + f2, next_order, left=left, right=right)
        heapq.heappush(heap, (parent.freq, next_order, parent))
        next_order += 1

    return heap[0][2]


def _code_lengths(node: _Node | None, depth: int, lengths: dict[Any, int]) -> None:
    if node is None:
        return
    if node.is_leaf:
        lengths[node.symbol] = max(depth, 1)
        return
    _code_lengths(node.left, depth + 1, lengths)
    _code_lengths(node.right, depth + 1, lengths)


def _canonical_table(lengths: dict[Any, int]) -> dict[Any, str]:
    if not lengths:
        return {}

    symbols = sorted(lengths.keys(), key=lambda symbol: (lengths[symbol], symbol))
    table: dict[Any, str] = {}
    code = 0
    prev_len = 0

    for symbol in symbols:
        length = lengths[symbol]
        code <<= length - prev_len
        table[symbol] = format(code, f"0{length}b")
        code += 1
        prev_len = length

    return table


def _bits_to_bytes(bits: list[int]) -> bytes:
    if not bits:
        return b""

    result = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        byte = 0
        for bit in chunk:
            byte = (byte << 1) | bit
        byte <<= 8 - len(chunk)
        result.append(byte)
    return bytes(result)


def canonical_huffman(data: list) -> tuple[bytes, dict[Any, str]]:
    """Canonical Huffman 符号化を行う。

    Args:
        data: 符号化対象のデータ列

    Returns:
        encoded: 符号化結果 (bytes)
        table: シンボル -> canonical Huffman code (bit string) の対応表
    """
    if not data:
        return b"", {}

    freq = Counter(data)
    root = _build_tree(freq)

    lengths: dict[Any, int] = {}
    _code_lengths(root, 0, lengths)
    table = _canonical_table(lengths)

    bits: list[int] = []
    for symbol in data:
        bits.extend(int(bit) for bit in table[symbol])

    return _bits_to_bytes(bits), table
