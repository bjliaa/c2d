import tensorflow as tf


class DiscreteNet(tf.keras.Model):
    """ Our network model for discrete distribution estimates """
    def __init__(self, num_actions, num_atoms, name):
        super(DiscreteNet, self).__init__(name=name)
        # DQN layers, we use the original padding of DQN for fewer weights.
        self.conv1 = tf.keras.layers.Conv2D(32,
                                            kernel_size=8,
                                            strides=4,
                                            input_shape=(4, 84, 84),
                                            data_format="channels_first")
        self.conv2 = tf.keras.layers.Conv2D(64,
                                            kernel_size=4,
                                            strides=2,
                                            data_format="channels_first")
        self.conv3 = tf.keras.layers.Conv2D(64,
                                            kernel_size=3,
                                            strides=1,
                                            data_format="channels_first")
        self.encoder = tf.keras.layers.Dense(512, activation="relu")

        self.logits = tf.keras.layers.Dense(
            num_actions * num_atoms,
            activation="linear",
            name="logits",
        )
        self.atom_embedder = tf.keras.layers.Dense(
            512,
            activation="relu",
            name="embedder",
        )
        self.atoms = tf.keras.layers.Dense(
            num_actions * num_atoms,
            activation="linear",
            name="atoms",
        )

        # Trainable parameter (alpha) for the support activation
        self.scale = tf.Variable(50.0)

        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, states, training=False):
        x = tf.cast(states, tf.float32) / 255.0

        # Representation network (DQN + BatchNormalization)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Flatten()(x)
        encoded_states = self.encoder(x)

        # Probability function
        logits = tf.reshape(self.logits(encoded_states), [-1, self.num_actions, self.num_atoms])
        probs = tf.keras.activations.softmax(logits, axis=-1)

        # Embedder and Atom function
        atoms_input = self.concat(encoded_states, probs, self.atom_embedder)
        supports = tf.reshape(self.atoms(atoms_input), [-1, self.num_actions, self.num_atoms])
        supports = self.scale * tf.tanh(supports / 5.0)  # Support activation
        return [probs, supports]

    def concat(self, encoded_states, x, embedder):
        y = tf.keras.layers.Flatten()(x)
        embedding = embedder(y)
        return tf.concat([encoded_states, embedding], axis=-1)
