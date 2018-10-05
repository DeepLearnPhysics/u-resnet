import numpy as np
import tables

NUM_ENTRIES = 10
DATA_SIZE = 100

data  = np.zeros(shape=(DATA_SIZE),dtype=np.float32)

FILTERS = tables.Filters(complib='zlib', complevel=5)
fout = tables.open_file('cacca.hdf5',mode='w', filters=FILTERS)
data_storage  = fout.create_earray(fout.root,'data',tables.Float32Atom(),shape=[0,DATA_SIZE])

for entry in range(NUM_ENTRIES):
    data[:] = 1.
    data_storage.append(data[None])
    
fout.close()
