import pandas as pd
import os

dopamine_games = [
    "airraid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bankheist",
    "battlezone",
    "beamrider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "choppercommand",
    "crazyclimber",
    "demonattack",
    "doubledunk",
    "elevatoraction",
    "enduro",
    "fishingderby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "icehockey",
    "jamesbond",
    "journeyescape",
    "kangaroo",
    "krull",
    "kungfumaster",
    "montezumarevenge",
    "mspacman",
    "namethisgame",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "privateeye",
    "qbert",
    "riverraid",
    "roadrunner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "spaceinvaders",
    "stargunner",
    "tennis",
    "timepilot",
    "tutankham",
    "upndown",
    "venture",
    "videopinball",
    "wizardofwor",
    "yarsrevenge",
    "zaxxon",
]

FOLDER = "experiments/new"

games_dict = {}
rom_list = os.listdir("ale_roms")
for rom_file in rom_list:
    rom = os.path.splitext(rom_file)[0]
    truncgame = rom.replace('_', '')
    if truncgame in dopamine_games:
        games_dict[rom] = truncgame


class Linear:
    def __init__(self, startval, endval, exploresteps):
        self.exp = exploresteps
        self.sval = startval
        self.endval = endval
        self.dydx = (endval - startval) / exploresteps

    def __call__(self, t):
        if t <= self.exp:
            return self.sval + t * self.dydx
        else:
            return self.endval


class ReturnFormatter:
    def __init__(self):
        self.last_str = ""

    def __call__(self, step, info):
        out_str = f"Step: {step}"
        for key in info:
            val = info[key]
            out_str += f", {key}: {val}"
        if len(out_str) > 100:
            out_str = out_str[:100]
        print(f"{out_str: <{len(self.last_str)}}", end="\r")
        self.last_str = out_str


def phase_formatter(iteration, episodes, avg_return, diff_time, steps):
    prefix_str = "TRAINING PHASE -> "
    steps_per_second = steps / diff_time
    ms_per_step = 1000 / steps_per_second
    print(
        prefix_str + f"Iteration: {iteration}, Episodes: {episodes}, " +
        f"Average return: {avg_return:.2f}, Steps per second: {steps_per_second:.1f} ( {ms_per_step:.2f} ms/step)"
    )


def loss_formatter(aloss, min_atom, max_atom):
    print(f"Avg Loss -> {aloss:.4f}")
    print(f"Max Theo. Supp Range -> [{min_atom:.2f}, {max_atom:.2f}]")


def makeRow(**kwargs):
    C = {}
    for key, value in kwargs.items():
        C[key] = value
    return C


def save_model(agent, dtag, game, folder=FOLDER):
    dopamine_game = games_dict[game]
    os.makedirs(f"{folder}/model_data/", exist_ok=True)
    prefix = f"{folder}/model_data/{dopamine_game}_{dtag}"
    agent.save_model(prefix)


def save_current_data(data_row_list, dtag, game, action_len, params, folder=FOLDER):
    dopamine_game = games_dict[game]
    os.makedirs(f"{folder}/training_data/", exist_ok=True)
    os.makedirs(f"{folder}/supplementary_data/", exist_ok=True)
    df = pd.DataFrame(data_row_list)
    df["Agent"] = "C2D"
    json_df = df[["iteration", "avg_return", "Agent"]]
    json_df = json_df.rename(columns={'iteration': 'Iteration', 'avg_return': 'Value'})
    json_df.to_json(f"{folder}/training_data/{dopamine_game}_{dtag}.json",
                    orient="records",
                    indent=2)
    df["tag"] = dtag
    df["game"] = dopamine_game
    df["actions"] = action_len
    for key in params:
        df[key] = params[key]
    df.to_csv(f"{folder}/supplementary_data/{dopamine_game}_{dtag}.csv")