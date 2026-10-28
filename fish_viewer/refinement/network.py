"""Voxel segmentation and boundary-weighted loss, with no radial shape constraint."""


def network(tf, config):
    layers = tf.keras.layers
    inputs = layers.Input((config.patch_size,)*3+(2,), name='membrane_and_seed')
    def block(value, filters):
        for _ in range(2):
            value = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(value)
            value = layers.LayerNormalization(axis=-1)(value)
            value = layers.Activation('relu')(value)
        return value
    value = inputs; skips = []
    for level in range(config.depth):
        value = block(value, config.filters*2**level)
        skips.append(value)
        value = layers.MaxPool3D(2)(value)
    value = block(value, config.filters*2**config.depth)
    for level in reversed(range(config.depth)):
        value = layers.UpSampling3D(2)(value)
        value = layers.Concatenate()([value,skips[level]])
        value = block(value, config.filters*2**level)
    logits = layers.Conv3D(1,1, dtype='float32', name='target_cell_logit')(value)
    return tf.keras.Model(inputs, logits, name='seeded_boundary_unet3d')


def loss(tf, target, logits, config=None):
    truth, weight, valid = target[...,0:1], target[...,1:2], target[...,2:3]
    probability = tf.math.sigmoid(logits)
    axes = (1,2,3,4)
    dice = (2*tf.reduce_sum(truth*probability*valid,axis=axes)+1) / (
        tf.reduce_sum((truth+probability)*valid,axis=axes)+1)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=logits)
    weight=weight*valid
    weighted = tf.reduce_sum(cross_entropy*weight,axis=axes)/tf.maximum(tf.reduce_sum(weight,axis=axes),1)
    region = 1-dice
    if config is not None and config.tversky_beta != .5:
        tp = tf.reduce_sum(truth*probability*valid,axis=axes)
        fn = tf.reduce_sum(truth*(1-probability)*valid,axis=axes)
        fp = tf.reduce_sum((1-truth)*probability*valid,axis=axes)
        region = 1-(tp+.5)/(tp+config.tversky_beta*fn+(1-config.tversky_beta)*fp+.5)
    value = region + .6*weighted
    if config is not None and config.surface_loss_weight:
        # Signed physical distance: negative inside, positive outside. Adding
        # the truth-dependent constant makes boundary loss nonnegative without
        # changing its gradients. Unknown/padded voxels have zero contribution.
        distance = target[...,3:4]
        boundary = tf.reduce_sum((probability-truth)*distance*valid,axis=axes) / tf.maximum(tf.reduce_sum(valid,axis=axes),1)
        value += config.surface_loss_weight*boundary
    return tf.reduce_mean(value), tf.reduce_mean(dice)
