#!/usr/bin/env python3
# Re-lay ffn_down of a dense model as n_ff / block "neuron-block experts" for block-sparse FFN (LLAMA_FFN_SPARSE_BLOCKS).
#
# blk.N.ffn_down [n_ff, n_embd] -> blk.N.ffn_down_blk [block, n_embd, n_ff / block]: block j of every row moves to expert j.
# Only whole quant blocks move, so the values are unchanged. ffn_gate / ffn_up are rows already and are not touched.
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-ffn-blockify")


def blockify(data: np.ndarray, qtype: gguf.GGMLQuantizationType, block: int) -> np.ndarray:
    # data: [n_embd, row_items], items are bytes for quantized types and values for float types
    blck_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if block % blck_size != 0:
        raise ValueError(f"block {block} is not a multiple of the {qtype.name} block size {blck_size}")
    n_embd, row_items = data.shape
    items_per_block = block // blck_size * type_size // data.dtype.itemsize
    n_blk = row_items // items_per_block
    if n_blk * items_per_block != row_items:
        raise ValueError(f"row of {row_items} items is not a whole number of {items_per_block}-item blocks")
    return np.ascontiguousarray(data.reshape(n_embd, n_blk, items_per_block).transpose(1, 0, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-lay ffn_down as neuron-block experts for block-sparse FFN")
    parser.add_argument("input", type=Path, help="input GGUF")
    parser.add_argument("output", type=Path, help="output GGUF")
    parser.add_argument("--block", type=int, default=256, help="neurons per block (multiple of the quant block size)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    reader = gguf.GGUFReader(args.input, 'r')
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_uint32(f"{arch}.ffn_block_size", args.block)

    # the MTP layer keeps its dense FFN
    mtp_layers = {t.name.split(".")[1] for t in reader.tensors if ".nextn." in t.name}

    tensors = []
    for t in reader.tensors:
        name, data = t.name, t.data
        if name.startswith("blk.") and name.endswith(".ffn_down.weight") and name.split(".")[1] not in mtp_layers:
            data = blockify(data, t.tensor_type, args.block)
            name = name.replace(".ffn_down.weight", ".ffn_down_blk.weight")
            logger.debug(f"{t.name} {t.tensor_type.name} {t.data.shape} -> {name} {data.shape}")
        tensors.append((name, data, t.tensor_type))
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing", total=sum(d.nbytes for _, d, _ in tensors), unit="byte", unit_scale=True)
    for _, data, _ in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        bar.update(data.nbytes)
    writer.close()


if __name__ == '__main__':
    main()
