import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as k

# k.set_floatx('float16')
# k.set_epsilon(1e-4)


class RecMF(keras.Model):

    def __init__(self, num_user, num_item, latent_dim):
        super(RecMF, self).__init__()

        self.num_user = num_user
        self.num_item = num_item

        self.user_rec = Bidirectional(GRU(latent_dim, kernel_regularizer=l2, bias_regularizer=l2))
        self.item_rec = Bidirectional(GRU(latent_dim, kernel_regularizer=l2, bias_regularizer=l2))

    def call(self, x, activate='softmax', training=None):

        user_index = x[:self.num_user]
        item_index = x[self.num_user:]

        user_embed = self.user_rec(user_index)
        item_embed = self.item_rec(item_index)

        output = tf.matmul(user_embed, item_embed)
        output = tf.keras.activations.sigmoid(output)

        return output
