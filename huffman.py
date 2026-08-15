"""
Canonical Huffman Code Generator
--------------------------------
データ(シンボルのリスト)から、カノニカル形式の
BITS(各符号長の個数) と HUFFVAL(符号長順に並べたシンボル) を生成する。

JPEGのDHTセグメントなどで使われている形式と同じ考え方です。
"""

from collections import Counter
import heapq


def build_code_lengths(data):
    """データからハフマン木を構築し、各シンボルの符号長(bit数)を求める"""
    freq = Counter(data)
    symbols = list(freq.keys())

    if len(symbols) == 1:
        # シンボルが1種類しかない場合は符号長1とする
        return {symbols[0]: 1}

    # (頻度, 挿入順, そのノードに含まれるシンボル一覧) のヒープ
    # 挿入順をキーに入れることで、リスト同士の比較エラーを防ぐ
    heap = []
    counter = 0
    for sym, f in freq.items():
        heapq.heappush(heap, (f, counter, [sym]))
        counter += 1

    length = {sym: 0 for sym in symbols}

    while len(heap) > 1:
        f1, _, syms1 = heapq.heappop(heap)
        f2, _, syms2 = heapq.heappop(heap)
        for s in syms1:
            length[s] += 1
        for s in syms2:
            length[s] += 1
        merged = syms1 + syms2
        heapq.heappush(heap, (f1 + f2, counter, merged))
        counter += 1

    return length


def lengths_to_bits_huffval(length_dict, max_bits=None):
    """符号長の辞書から BITS と HUFFVAL を作る(カノニカル順序に並べ替え)"""
    # (符号長, シンボル値) でソート -> これがカノニカルの順序
    sorted_syms = sorted(length_dict.items(), key=lambda x: (x[1], x[0]))

    max_len = max(l for _, l in sorted_syms)
    if max_bits is not None and max_len > max_bits:
        raise ValueError(f"符号長が上限({max_bits}bit)を超えています: {max_len}bit")

    size = (max_bits or max_len) + 1  # index 0 は使わない
    bits = [0] * size
    huffval = []

    for sym, l in sorted_syms:
        bits[l] += 1
        huffval.append(sym)

    return bits[1:], huffval  # 1bit目から数える形にして返す


def canonical_huffman(data, max_bits=None):
    """メイン関数: データ(リスト)から BITS, HUFFVAL を返す"""
    lengths = build_code_lengths(data)
    bits, huffval = lengths_to_bits_huffval(lengths, max_bits=max_bits)
    return bits, huffval


def assign_canonical_codes(bits, huffval):
    """(おまけ) BITSとHUFFVALから、実際の2進符号を割り当てて確認する"""
    codes = {}
    code = 0
    val_idx = 0
    for length, count in enumerate(bits, start=1):
        for _ in range(count):
            codes[huffval[val_idx]] = format(code, f"0{length}b")
            code += 1
            val_idx += 1
        code <<= 1
    return codes


def encode(data, bits, huffval):
    """データ(シンボルのリスト)を BITS/HUFFVAL の符号表でビット列(文字列)に変換する"""
    codes = assign_canonical_codes(bits, huffval)
    try:
        return "".join(codes[sym] for sym in data)
    except KeyError as e:
        raise ValueError(f"符号表にないシンボルです: {e.args[0]!r}") from e


def decode(bitstring, bits, huffval):
    """ビット列(文字列)を BITS/HUFFVAL の符号表でデータ(シンボルのリスト)に戻す"""
    codes = assign_canonical_codes(bits, huffval)
    code_to_sym = {code: sym for sym, code in codes.items()}

    result = []
    buf = ""
    for ch in bitstring:
        buf += ch
        if buf in code_to_sym:
            result.append(code_to_sym[buf])
            buf = ""

    if buf:
        raise ValueError(f"末尾に不完全な符号が残っています: {buf!r}")

    return result


def pack_bits(bitstring):
    """'0'/'1'の文字列を実際のbytesに詰める。末尾は0でパディングする。
    戻り値: (packed_bytes, 元のビット数)
    """
    pad = (-len(bitstring)) % 8
    padded = bitstring + "0" * pad
    packed = bytes(
        int(padded[i:i + 8], 2) for i in range(0, len(padded), 8)
    )
    return packed, len(bitstring)


def unpack_bits(packed, num_bits):
    """pack_bits で作った bytes を、元のビット数だけ '0'/'1' の文字列に戻す"""
    bitstring = "".join(format(byte, "08b") for byte in packed)
    return bitstring[:num_bits]


if __name__ == "__main__":
    data = list("this is an example of a huffman tree")

    bits, huffval = canonical_huffman(data)
    print("BITS   :", bits)
    print("HUFFVAL:", huffval)

    print("\n--- 確認用: 実際の符号 ---")
    codes = assign_canonical_codes(bits, huffval)
    for sym, code in sorted(codes.items(), key=lambda x: (len(x[1]), x[0])):
        print(f"  {sym!r}: {code}")

    print("\n--- エンコード / デコード ---")
    encoded = encode(data, bits, huffval)
    print("ビット列  :", encoded)
    print("エンコード後のbit数 :", len(encoded), " / 元データ長 x 8 =", len(data) * 8)

    decoded = decode(encoded, bits, huffval)
    print("デコード結果が一致するか:", decoded == data)

    print("\n--- バイト列への詰め込み ---")
    packed, num_bits = pack_bits(encoded)
    print("バイト数:", len(packed), "(元のビット数:", num_bits, ")")

    restored_bits = unpack_bits(packed, num_bits)
    restored_data = decode(restored_bits, bits, huffval)
    print("バイト経由でも一致するか:", restored_data == data)