import sys
import numpy as np

wav_header_dtype = np.dtype([
     ("chunk_id", (str, 4)),   # flexible-sized scalar type, item size 4
     ("chunk_size", "<u4"),    # little-endian unsigned 32-bit integer
     ("format", "S4"),         # 4-byte string
     ("fmt_id", "S4"),
     ("fmt_size", "<u4"),
     ("audio_fmt", "<u2"),     #
     ("num_channels", "<u2"),  # .. more of the same ...
     ("sample_rate", "<u4"),   #
     ("byte_rate", "<u4"),
     ("block_align", "<u2"),
     ("bits_per_sample", "<u2"),
     ("data_id", ("S1", (2, 2))), # sub-array! **MUST** be fixed-size
     ("data_size", "u4"),
     #
     # the sound data itself cannot be represented here:
     # it does not have a fixed size
])

print wav_header_dtype.fields

# Mini-exercise: Rewrite the above by supplying only the ``sample_rate`` and 
#                ``num_channels`` fields.
#
#  wav_header_dtype = np.dtype(dict(
#      names=['format', 'sample_rate', 'data_id'],
#      offsets= list of offsets in bytes, from start of structure
#      formats= list of dtypes for each field,
#  ))


f = open(sys.argv[1], 'r')
wav_header = np.fromfile(f, dtype=wav_header_dtype, count=1)
f.close()

print "Sample rate: %d, channels: %d" % (
    wav_header['sample_rate'][0],
    wav_header['num_channels'][0]
    )
