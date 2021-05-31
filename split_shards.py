import tensorflow as tf

def custom_reader_func(datasets: tf.data.Dataset):
    datasets = datasets.shuffle(20)
    return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=20)

ds = tf.data.experimental.load('/home/alex/full_ds', (
                            tf.TensorSpec(shape=[678400], dtype=tf.float32), 
                            tf.TensorSpec(shape=[16], dtype=tf.float32), 
                            tf.TensorSpec(shape=[64], dtype=tf.float32), 
                            tf.TensorSpec(shape=None, dtype=tf.float32),
                            tf.TensorSpec(shape=None, dtype=tf.float32),
                            tf.TensorSpec(shape=None, dtype=tf.float32),
                            tf.TensorSpec(shape=[64, 2650, 3], dtype=tf.float32)), compression='GZIP', reader_func=custom_reader_func)

for i in range(20):
    shard = ds.take(7904)
    shard = shard.shuffle(7904)
    tf.data.experimental.save(shard, '/home/alex/dataset-drive/full_ds_sharded/shard_%d'%i, compression='GZIP')
    ds = ds.skip(7904)
# options = tf.data.Options()
# options.experimental_threading.private_threadpool_size = 1
# ds = ds.with_options(options)

# def shard_func(index, data):
#     return tf.constant(1, tf.int64)

# tf.data.experimental.save(ds, '/home/alex/alex-usbb/interleaved_shuffled', compression='GZIP')
# print(enumerated.element_spec)
# print(next(iter(enumerated)))