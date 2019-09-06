import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as k

# k.set_floatx('float16')
# k.set_epsilon(1e-4)


class RecMF(keras.Model):

    def __init__(self, num_user, num_item, latent_dim, seq_len):
        super(RecMF, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.seq_len = seq_len

        self.user_rec = Bidirectional(GRU(latent_dim, kernel_regularizer=l2, bias_regularizer=l2))
        self.item_rec = Bidirectional(GRU(latent_dim, kernel_regularizer=l2, bias_regularizer=l2))

    def call(self, x, activate='softmax', training=None):

        user_one_hot = x[:, :, :self.num_user]
        item_one_hot = x[:, :, self.num_user:]

        user_embed = self.user_rec(user_one_hot)
        item_embed = self.item_rec(item_one_hot)

        output = tf.reduce_sum(tf.multiply(user_embed, item_embed))
        output_logit = tf.keras.activations.sigmoid(output)

        return output_logit

    def get_user_weight(self):
        forward_kernel = [v for v in self.user_rec.weights
                          if v.name == "rec_mf/bidirectional/forward_gru/kernel:0"][0]
        forward_rec_kernel = [v for v in self.user_rec.weights
                              if v.name == "rec_mf/bidirectional/forward_gru/recurrent_kernel:0"][0]
        forward_bias = [v for v in self.user_rec.weights if v.name == "rec_mf/bidirectional/forward_gru/bias:0"][0]

        backward_kernel = [v for v in self.user_rec.weights
                           if v.name == "rec_mf/bidirectional/backward_gru/kernel:0"][0]
        backward_rec_kernel = [v for v in self.user_rec.weights
                               if v.name == "rec_mf/bidirectional/backward_gru/recurrent_kerneforward_kernel.numpy()l:0"][0]
        backward_bias = [v for v in self.user_rec.weights if v.name == "rec_mf/bidirectional/backward_gru/bias:0"][0]

        return forward_kernel, forward_rec_kernel, forward_bias, backward_kernel, backward_rec_kernel, backward_bias

    def infer_success_exposure_count(self, uid, iid, threshold):
        x = tf.concat([tf.one_hot(uid, self.num_user, dtype=tf.float32),
                       tf.one_hot(iid, self.num_item, dtype=tf.float32)], axis=1)
        x = tf.tile(tf.expand_dims(x, axis=1), [1, self.seq_len, 1])
        output_logit = self(x)

        num_passed_quantum = 0
        while output_logit[num_passed_quantum] > threshold:
            num_passed_quantum += 1

        return num_passed_quantum, output_logit[num_passed_quantum+1]

    def infer_user_consumption(self, uid):
        user_forward_kernel = [v for v in self.user_rec.weights
                               if v.name == "rec_mf/bidirectional/forward_gru/kernel:0"][0].numpy()
        user_backward_kernel = [v for v in self.user_rec.weights
                                if v.name == "rec_mf/bidirectional/backward_gru/kernel:0"][0].numpy()
        vec_user_embed = user_forward_kernel[uid, :20] + user_backward_kernel[uid, :20]

        item_forward_kernel = [v for v in self.item_rec.weights
                               if v.name == "rec_mf/bidirectional_1/forward_gru_1/kernel:0"][0].numpy()
        item_backward_kernel = [v for v in self.item_rec.weights
                                if v.name == "rec_mf/bidirectional_1/backward_gru_1/kernel:0"][0].numpy()
        mat_item_embeds = item_forward_kernel[:, :20] + item_backward_kernel[:, :20]

        return tf.matmul(mat_item_embeds, tf.expand_dims(vec_user_embed, axis=1))

