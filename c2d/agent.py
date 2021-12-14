import tensorflow as tf

from c2d.models import DiscreteNet


class Agent(tf.Module):
    """ C2D agent class to handle action selection and learning algorithms """
    def __init__(self, action_len, starteps, gamma, learning_rate, adameps, evaleps, heps, atoms):
        super(Agent, self).__init__(name="Agent")

        self.eps = tf.Variable(starteps, trainable=False, name="epsilon")
        self.h_eps = tf.constant(heps, dtype=tf.float32)
        self.evaleps = tf.constant(evaleps, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.action_len = tf.constant(action_len, dtype=tf.int64)
        self.suppsize = atoms

        self.opt = tf.optimizers.Adam(learning_rate=learning_rate[0], epsilon=adameps)
        self.net = DiscreteNet(num_actions=action_len, num_atoms=self.suppsize, name="AgentNet")

        # Clone network for distributional DQN
        self.tnet = DiscreteNet(num_actions=action_len, num_atoms=self.suppsize, name="TargetNet")

    @tf.function
    def eps_greedy_action(self, states, epsval):
        """ Selects epsilon greedy actions given current estimations and epsilon value """
        self.eps.assign(epsval)
        dice = (tf.random.uniform([tf.shape(states)[0]], minval=0, maxval=1, dtype=tf.float32) <
                self.eps)
        raction = tf.random.uniform(
            [tf.shape(states)[0]],
            minval=0,
            maxval=self.action_len,
            dtype=tf.int64,
        )
        qaction = tf.argmax(self.qvalues(states), axis=-1)
        return tf.cast(tf.where(dice, raction, qaction), tf.uint8)

    @tf.function
    def qvalues(self, states):
        """ Computes Q-values for the given states in a way that respects transformations by phi """
        probs, supps = self.net(states)
        return tf.einsum("ajk, ajk-> aj", probs, self.phiinv(supps))

    @tf.function
    def target_estimates(self, states):
        probs, supps = self.tnet(states, training=True)
        return probs, supps

    @tf.function
    def update_target(self):
        """ Copies the current network to the target network """
        avars = self.net.trainable_variables
        tvars = self.tnet.trainable_variables
        for w1, w2 in zip(avars, tvars):
            w2.assign(w1)

    @tf.function
    def train(self, states, actions, rewards, end_states, dones):
        """ Handles all learning, i.e., updates weights by distributional DQN with a proper Cramér loss """
        # Watch trainable variables during loss computation
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(self.net.trainable_variables)
            batch_size = tf.shape(states)[0]
            brange = tf.range(0, batch_size)
            indices = tf.stack([brange, tf.cast(actions, tf.int32)], axis=1)

            # Current estimations over all actions
            probs, supps = self.net(states, training=True)

            # Select by sampled actions
            estimated_probs = tf.gather_nd(probs, indices)
            estimated_supps = tf.gather_nd(supps, indices)

            # Target estimations
            end_probs, end_supps = self.target_estimates(end_states)
            tvalues = tf.einsum("ajk, ajk-> aj", end_probs, self.phiinv(end_supps))
            end_actions = tf.argmax(tvalues, axis=1, output_type=tf.int32)
            end_indices = tf.stack([brange, end_actions], axis=1)
            argmax_probs = tf.gather_nd(end_probs, end_indices)
            argmax_supps = tf.gather_nd(end_supps, end_indices)

            #  --- Pushforward estimation with observed reward. ---
            # If an item is terminal then all pushed atoms will equal the reward,
            # which effectively creates a single delta regardless of probability vector.
            not_done_mask = tf.reshape((1.0 - dones), shape=(batch_size, 1))
            rews = tf.reshape(rewards, shape=(batch_size, 1))
            target_probs = argmax_probs
            target_supps = self.phi(rews + self.gamma * not_done_mask * self.phiinv(argmax_supps))

            # Our loss is the Cramér distance
            losses = self.cramer_distance(target_probs, target_supps, estimated_probs,
                                          estimated_supps)
            loss = tf.reduce_mean(losses)

        # Adjust weights by gradient so as to minimize the Cramér distance
        grads = tape.gradient(loss, self.net.trainable_variables)

        # Since the Cramér distance is sensitive to the underlying geometry,
        # we clip gradients by the global norm for stability in environments with very large returns.
        # It is clear that clipping by the global norm is the correct way to preserve gradient directions.
        global_norm = tf.linalg.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, 10.0, use_norm=global_norm)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))

        min_atom = tf.reduce_min(estimated_supps)
        max_atom = tf.reduce_max(estimated_supps)
        norm = global_norm
        return loss, min_atom, max_atom, norm

    def phi(self, x, scale=1.99):
        """ Homeomorphism """
        return scale * (tf.math.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + self.h_eps * x)

    def phiinv(self, y, scale=1.99):
        """ Inverse homeomorphism """
        num = tf.sqrt(1 + 4 * self.h_eps * (tf.abs(y / scale) + 1 + self.h_eps)) - 1
        den = 2 * self.h_eps
        return tf.math.sign(y) * ((num / den)**2 - 1)

    @tf.function
    def cramer_distance(self, target_probs, target_supps, estimated_probs, estimated_supps):
        """ Computes the Cramér distance between two distributions mu, nu by computing the L2-norm squared of the CDF for the signed measure mu - nu """
        batchsize = target_probs.shape[0]

        # Concatenate and sort the extended support for the signed measure of each batch item
        z_extended = tf.concat([target_supps, estimated_supps], axis=-1)
        suppsize_e = z_extended.shape[-1]
        idx = tf.argsort(z_extended, axis=-1)
        batch_indices = tf.constant([[b for _ in range(suppsize_e)] for b in range(batchsize)])
        batch_indices = tf.stack([batch_indices, idx], axis=-1)
        z_extended = tf.gather_nd(z_extended, batch_indices)

        # Compute the relevant deltas for the integrals
        z_deltas = z_extended[:, 1:] - z_extended[:, :-1]

        # Compute signed probabilities given the extended support for each batch item
        signed_measure_mass = tf.concat([target_probs, -estimated_probs], axis=-1)
        signed_measure_mass = tf.gather_nd(signed_measure_mass, batch_indices)

        # Our loss is then the L2-norm squared of the CDF
        integrands = tf.cumsum(signed_measure_mass, axis=-1)**2
        losses = tf.reduce_sum(integrands[:, :-1] * z_deltas, axis=-1)
        return losses

    def save_model(self, prefix):
        net_str = prefix + "_dnet.h5"
        self.net.save_weights(net_str)
