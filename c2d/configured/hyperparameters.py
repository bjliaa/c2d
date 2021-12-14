def paramdict_single():

    d = {
        "explore steps": int(250e3),
        "start epsilon": 1.0,
        "final epsilon": 0.01,
        "target update period": int(8e3),
        "gamma": 0.99,
        "learning rate": 0.5e-4,
        "adam epsilon": 0.01/32,
        "mem size": int(1e6),
        "prefill size": int(20e3),
        "max episode steps": 27000,
        "intensity": 8,
        "repeat action probability": 0.25,
        "batch size": 32,
        "batch prefetch size": 8,
        "training phase steps": 250000,
        "eval phase steps": 0,
        "evaluation epsilon": 0.001,
        "scaling epsilon": 0.001,
        "atoms": 32,
    }
    return d
