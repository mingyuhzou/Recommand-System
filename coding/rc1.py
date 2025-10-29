import tensorflow as tf
from tensorflow.keras.layers import StringLookup
data = [["a", "c", "d"], ["d", "z", "b"]]
layer = StringLookup()
layer.adapt(data)

print(layer.get_vocabulary())
