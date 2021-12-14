import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

batch_size = 32
prefetch_batch_size = 8
obs_stack = 4
obs_width = 84

lt = {
    "env": ctypes.c_void_p,
    "game": ctypes.c_char_p,
    "mem size": ctypes.c_int,
    "stateBuffer": ndpointer(
        dtype=np.uint8,
        shape=(1, obs_stack, obs_width, obs_width),
        flags=["C", "O", "W", "A"],
    ),
    "action": ctypes.c_uint8,
    "reward": ctypes.c_int,
    "bs": ndpointer(
        dtype=np.uint8,
        shape=(batch_size, obs_stack, obs_width, obs_width),
        flags=["C", "O", "W", "A"],
    ),
    "ba": ndpointer(
        dtype=np.uint8, shape=batch_size, flags=["C", "O", "W", "A"]
    ),
    "br": ndpointer(
        dtype=np.float32, shape=batch_size, flags=["C", "O", "W", "A"]
    ),
    "bes": ndpointer(
        dtype=np.uint8,
        shape=(batch_size, obs_stack, obs_width, obs_width),
        flags=["C", "O", "W", "A"],
    ),
    "bd": ndpointer(
        dtype=np.float32, shape=batch_size, flags=["C", "O", "W", "A"]
    ),
    "num actions": ctypes.c_int,
    "batch position": ctypes.c_int,
    "batch count": ctypes.c_int,
    "episode steps": ctypes.c_int,
    "episode score": ctypes.c_int,
    "rgb dims": ctypes.c_int,
    "rgb": ndpointer(
        dtype=np.uint8, flags=["C", "O", "W", "A"]
    )
}

libc2d = ctypes.cdll.LoadLibrary("libc2d.so")

libc2d.newEnv.argtypes = None
libc2d.newEnv.restype = lt["env"]
libc2d.delEnv.argtypes = [lt["env"]]
libc2d.delEnv.restype = None
libc2d.initEnv.argtypes = [
    lt["env"],
    lt["game"],
    lt["mem size"],
]
libc2d.initEnv.restype = None
libc2d.resetEnv.argtypes = [lt["env"]]
libc2d.resetEnv.restype = None
libc2d.hardResetEnv.argtypes = [lt["env"]]
libc2d.hardResetEnv.restype = None
libc2d.getObsEnv.argtypes = [lt["env"], lt["stateBuffer"]]
libc2d.getObsEnv.restype = None
libc2d.actionLengthEnv.argtypes = [lt["env"]]
libc2d.actionLengthEnv.restype = lt["num actions"]
libc2d.episodeStepsEnv.argtypes = [lt["env"]]
libc2d.episodeStepsEnv.restype = lt["episode steps"]
libc2d.episodeStepsEnv.argtypes = [lt["env"]]
libc2d.episodeScoreEnv.restype = lt["episode score"]
libc2d.actEnv.argtypes = [lt["env"], lt["action"], ctypes.c_bool]
libc2d.actEnv.restype = lt["reward"]
libc2d.gameOverEnv.argtypes = [lt["env"]]
libc2d.gameOverEnv.restype = ctypes.c_bool
libc2d.maxStepReachedEnv.argtypes = [lt["env"]]
libc2d.maxStepReachedEnv.restype = ctypes.c_bool
libc2d.doneEnv.argtypes = [lt["env"]]
libc2d.doneEnv.restype = ctypes.c_bool
libc2d.prefetchBatchEnv.argtypes = [
    lt["env"],
    lt["bs"],
    lt["ba"],
    lt["br"],
    lt["bes"],
    lt["bd"],
    lt["batch position"],
    lt["batch count"],
]
libc2d.prefetchBatchEnv.restype = None
libc2d.getRawWidthEnv.argtypes = [lt["env"]]
libc2d.getRawWidthEnv.restype = lt["rgb dims"]
libc2d.getRawHeightEnv.argtypes = [lt["env"]]
libc2d.getRawHeightEnv.restype = lt["rgb dims"]
libc2d.getRGBEnv.argtypes = [lt["env"], lt["rgb"]]
libc2d.getRGBEnv.restype = None

class AtariEnv:
    def __init__(
        self, game, mem_size,
    ):
        gamestr = f"ale_roms/{game}.bin"
        self.env_p = libc2d.newEnv()
        libc2d.initEnv(
            self.env_p, gamestr.encode("utf-8"), mem_size,
        )
        self.rawWidth = libc2d.getRawWidthEnv(self.env_p)
        self.rawHeight = libc2d.getRawHeightEnv(self.env_p)
        rawsize = self.rawHeight*self.rawWidth*3
        self.rawBuffer = np.zeros(rawsize, dtype=np.uint8)
        self.batch_it = 0
        self.stateBuffer = np.zeros(
            (1, obs_stack, obs_width, obs_width), dtype=np.uint8
        )
        self.bs = np.zeros(
            (batch_size, obs_stack, obs_width, obs_width), dtype=np.uint8
        )
        self.ba = np.zeros(batch_size, dtype=np.uint8)
        self.br = np.zeros(batch_size, dtype=np.float32)
        self.bes = np.zeros(
            (batch_size, obs_stack, obs_width, obs_width), dtype=np.uint8
        )
        self.bd = np.zeros(batch_size, dtype=np.float32)

    def step(self, action, batchWork, evalmode=False):
        self._act(action, evalmode)
        d = self._done()
        terminal = self._gameOver() or self._maxStepReached()
        info = {}
        if d:
            self._reset()
        if terminal:
            info["Episode Length"] = self._episodeSteps()
            info["Episode Score"] = self._episodeScore() 
        if batchWork:
            self._prefetch(self.batch_it)
            self.batch_it = (self.batch_it + prefetch_batch_size) % batch_size
        self._updateObs()
        return self.stateBuffer, terminal, info

    def actionLength(self):
        return libc2d.actionLengthEnv(self.env_p)

    def getObs(self):
        self._updateObs()
        return self.stateBuffer

    def getBatch(self):
        return self.bs, self.ba, self.br, self.bes, self.bd

    def hardReset(self):
        self.batch_it = 0
        libc2d.hardResetEnv(self.env_p)
    
    def getRGB(self):
        libc2d.getRGBEnv(self.env_p, self.rawBuffer)
        return np.reshape(self.rawBuffer, (self.rawHeight,self.rawWidth, 3))

    def _prefetch(self, b_idx):
        libc2d.prefetchBatchEnv(
            self.env_p,
            self.bs,
            self.ba,
            self.br,
            self.bes,
            self.bd,
            b_idx,
            prefetch_batch_size,
        )

    def _updateObs(self):
        libc2d.getObsEnv(self.env_p, self.stateBuffer)

    def _act(self, action, evalmode):
        return libc2d.actEnv(self.env_p, action, evalmode)

    def _gameOver(self):
        return libc2d.gameOverEnv(self.env_p)

    def _maxStepReached(self):
        return libc2d.maxStepReachedEnv(self.env_p)
    
    def _episodeSteps(self):
        return libc2d.episodeStepsEnv(self.env_p)

    def _episodeScore(self):
        return libc2d.episodeScoreEnv(self.env_p)

    def _done(self):
        return libc2d.doneEnv(self.env_p)

    def _reset(self):
        libc2d.resetEnv(self.env_p)
    
    def _noopStart(self):
        libc2d.noopStartEnv(self.env_p) 
