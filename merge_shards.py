import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_control_flow_ops import merge

# merged_ds = tf.data.experimental.load('/home/alex/alex-usb/compressed_ds_shards/shard_0', (
#                         tf.TensorSpec(shape=[678400], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[16], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[64], dtype=tf.float32), 
#                         tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')
    
datasets = []
for i in range(0, 20):
    ds = tf.data.experimental.load('/home/alex/dataset-drive/ds_sharded/shard_%d'%i, (
                            tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                            tf.TensorSpec(shape=[16], dtype=tf.float32), 
                            tf.TensorSpec(shape=[64], dtype=tf.float32), 
                            tf.TensorSpec(shape=None, dtype=tf.float32)), compression='GZIP')
    datasets.append(ds)

# merged_ds = tf.data.experimental.sample_from_datasets(datasets)
# print(len(merged_ds))
def shard_func(w, x, y, z):
    return tf.random.uniform([1], minval=0, maxval=20, dtype=tf.int64)

# merged_ds = tf.data.experimental.sample_from_datasets(datasets)


tf.data.experimental.save(merged_ds, '/home/alex/dataset-drive/ds_merged', shard_func=shard_func)
