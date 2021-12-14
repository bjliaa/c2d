import numpy as np
import tensorflow as tf
import numpy as np
import time
from c2d.agent import Agent
from c2d.util import (Linear, ReturnFormatter, phase_formatter, loss_formatter, makeRow, save_model,
                      save_current_data)
from c2d.configured.hyperparameters import paramdict_single
from c2d.configured.atarienv import AtariEnv


class Runner:
    """ Simple class to handle experiments. This includes agent exploration & training in atari MDPs """
    def __init__(self, game, iterations, dtag):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.game = game
        self.iterations = iterations
        self.dtag = dtag
        self.params = paramdict_single()
        self.training_steps = self.params["training phase steps"]
        self.eval_steps = self.params["eval phase steps"]
        self.prefill_history = self.params["prefill size"]
        self.batch_size = self.params["batch size"]
        self.prefetch_size = self.params["batch prefetch size"]
        self.train_update_period = self.batch_size // self.prefetch_size
        self.target_update_period = self.params["target update period"]
        self.eps = Linear(
            self.params["start epsilon"],
            self.params["final epsilon"],
            self.params["explore steps"],
        )
        assert self.train_update_period == self.batch_size / self.prefetch_size

        print("Creating environment/memory/network...")
        self.env = AtariEnv(game, mem_size=self.params["mem size"])
        self.action_len = self.env.actionLength()
        self.agent = Agent(
            self.action_len,
            starteps=self.params["start epsilon"],
            gamma=self.params["gamma"],
            learning_rate=[self.params["learning rate"], self.params["learning rate"]],
            adameps=self.params["adam epsilon"],
            evaleps=self.params["evaluation epsilon"],
            heps=self.params["scaling epsilon"],
            atoms=self.params["atoms"])
        print("Done.")
        self.return_formatter = ReturnFormatter()

    def run(self):
        """ One full experiment run for a number of iterations measured in 1M frames. 
            Collects and stores statistics, saves models. """
        print("Collecting random history...")
        self._prefill()
        self._warmup_construct()
        self._output_settings()
        print("Waiting for initial returns...", end="\r")
        tottime = time.perf_counter()
        data_row_list = []
        for iteration in range(self.iterations):
            diff_time, episodes, avg_return, avg_loss, min_atom, max_atom, gnorm = self._train_phase(
                iteration)
            phase_formatter(iteration, episodes, avg_return, diff_time, self.training_steps)
            loss_formatter(avg_loss, min_atom, max_atom)
            data_row_list.append(
                makeRow(iteration=iteration,
                        total_steps=(iteration + 1) * self.training_steps,
                        episodes=episodes,
                        avg_return=avg_return,
                        avg_loss=avg_loss,
                        supp_min=min_atom,
                        supp_max=max_atom,
                        norm_max=gnorm))
            print("=" * 64)
            save_current_data(data_row_list, self.dtag, self.game, self.action_len, self.params)

        save_model(self.agent, self.dtag, self.game)

        diff_time = time.perf_counter() - tottime
        print(f"Learning done in {diff_time}s.")
        print("Done.")

    def _output_settings(self):
        print("=" * 24 + " Settings Agent " + "=" * 24)
        print(f"Data Tag: {self.dtag}")
        print(f"Game: {self.game}")
        print(f"Actions: {self.action_len}")
        for key in self.params:
            print(f"{key}: {self.params[key]}")
        print("Learning...")
        print("=" * 64)

    def _prefill(self):
        """ Prefills the replay buffer with random history """
        start = time.perf_counter()
        bwork = False
        for t in range(1, self.prefill_history + 1):
            action = np.random.randint(self.action_len, dtype=np.uint8)
            if t == (self.prefill_history - self.batch_size + 1):
                bwork = True
            self.env.step(action, batchWork=bwork)
            if t % 10000 == 0:
                tf.print(f"Collected {t} samples.")

        mssmp = 1000 * (time.perf_counter() - start) / self.prefill_history
        self.env.hardReset()
        print(f"Done ({mssmp:.2f} ms/smp).")

    def _warmup_construct(self):
        states, _, _, _, _ = self.env.getBatch()
        self.agent.qvalues(states)
        self.agent.target_estimates(states)
        self.agent.update_target()

    def _train_phase(self, iteration):
        """ One iteration training loop 250k steps (1M frames) """
        state = self.env.getObs()
        phase_time = time.perf_counter()
        train_scores = []
        losses = []
        min_atom, max_atom = (np.finfo(np.float32).max, np.finfo(np.float32).min)
        max_norm = np.finfo(np.float32).min
        # Exploration and training loop
        for train_step in range(self.training_steps):
            # Update current epsilon (only relevant on the first training phase)
            current_step = iteration * self.training_steps + train_step
            t_eps = tf.constant(self.eps(current_step), dtype=tf.float32)

            # Compute eps-greedy action given the current state
            action = (self.agent.eps_greedy_action(state, t_eps).numpy().astype(np.uint8))[0]

            # Take step by action
            with tf.device("/CPU:0"):
                state, terminal, info = self.env.step(action, batchWork=True)
                if terminal:
                    train_scores.append(info["Episode Score"])
                    self.return_formatter(current_step + 1, info)

            # Periodically train (every 4th step)
            if train_step % self.train_update_period == 0:
                with tf.device("/CPU:0"):
                    sts, acs, rws, ests, dns = self.env.getBatch()
                loss, amin, amax, norm = self.agent.train(sts, acs, rws, ests, dns)
                with tf.device("/CPU:0"):
                    losses.append(loss)
                    min_atom = tf.minimum(min_atom, amin).numpy()
                    max_atom = tf.maximum(max_atom, amax).numpy()
                    max_norm = tf.maximum(max_norm, norm).numpy()

            # Periodically update the clone network for distributional DQN (every 8k steps)
            if current_step % self.target_update_period == 0:
                self.agent.update_target()

        diff_time = time.perf_counter() - phase_time
        episodes = len(train_scores)
        avg_return = np.nan if episodes == 0 else np.mean(np.array(train_scores))
        avg_loss = np.nan if episodes == 0 else np.mean(np.array(losses))
        return diff_time, episodes, avg_return, avg_loss, min_atom, max_atom, max_norm