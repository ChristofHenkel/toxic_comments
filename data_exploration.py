from utils import *

def show_hist(meta_data):
    meta_data.hist()


meta_data = read_metadata(raw_data_dir,train_fn)

a = [sum(item['label']) for item in corpus]
len([item for item in a if item == 0])
86061
len([item for item in a if item == 1])
3833
len([item for item in a if item == 2])
2107
len([item for item in a if item == 3])
2523
len([item for item in a if item == 4])
1076
len([item for item in a if item == 5])
231
