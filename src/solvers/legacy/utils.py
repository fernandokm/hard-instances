import struct

import numpy as np


def encode_array(arr: np.ndarray) -> bytes:
    dtype = str(arr.dtype).encode("utf-8")

    preamble_format = "<10sB" + "I" * len(arr.shape)
    preamble = struct.pack(preamble_format, dtype, len(arr.shape), *arr.shape)

    encoded_dtype = arr.dtype.newbyteorder("<")
    payload = arr.astype(encoded_dtype).tobytes()

    return preamble + payload


def decode_array(b: bytes) -> np.ndarray:
    dtype_len = 10
    shape_len = b[dtype_len]
    preamble_format = f"<{dtype_len}sB" + "I" * shape_len
    preamble_size = shape_len * 4 + dtype_len + 1
    dtype, _, *shape = struct.unpack(preamble_format, b[:preamble_size])

    dtype = np.dtype(dtype.decode("utf-8").rstrip("\x00"))
    encoded_dtype = dtype.newbyteorder("<")

    arr_flat = np.frombuffer(b[preamble_size:], dtype=encoded_dtype)
    arr = arr_flat.reshape(shape).astype(dtype)
    return arr
