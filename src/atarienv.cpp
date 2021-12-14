#include "atarienv.hpp"
#include "common.hpp"
#include <algorithm>
#include <iterator>
#include <opencv2/imgproc.hpp>

namespace c2d {

void AtariEnv::initialize(const std::string &game, int memSize) {
  memory = std::make_unique<ReplayBuffer>(memSize);
  ale_interface.setInt("random_seed", time(0));
  ale_interface.setFloat("repeat_action_probability", repeatActionProbability);
  ale_interface.loadROM(game);
  rawWidth = static_cast<int>(ale_interface.getScreen().width());
  rawHeight = static_cast<int>(ale_interface.getScreen().height());
  rawFrames.at(0).resize(rawHeight * rawWidth, 0);
  rawFrames.at(1).resize(rawHeight * rawWidth, 0);
  obsWin.resize(stateSize, 0);  
  actionMap = ale_interface.getMinimalActionSet();
  numActions = static_cast<int>(actionMap.size());
  resetObs();
}

auto AtariEnv::getActionLength() const -> int { return numActions; }

auto AtariEnv::act(action_t action, bool evalmode) -> reward_t {
  auto rew = act_func(action);
  if (!evalmode) {
    FlatState endstate(stateSize);
    getObs(endstate);
    if (!framesReached) { // Don't store and train on max step terminations
      memory->add(action, rew, endstate, episodeDone);
    }
  }
  return rew;
}

auto AtariEnv::act(action_t action, std::span<pixel_t> sbuff) -> reward_t {
  auto rew = act_func(action);
  getObs(sbuff);
  if (!framesReached) { // Don't store and train on max step terminations
    memory->add(action, rew, sbuff, episodeDone);
  }
  return rew;
}

auto AtariEnv::act_func(action_t action) -> reward_t {
  ++stepCount;
  framesReached = stepCount == endStepMax;
  auto a = actionMap[action];
  auto prelives = ale_interface.lives();
  reward_t rew = 0.0;
  for (int t = 0; t < frameSkip; t++) { // Loop ALE by Dopamine logic
    auto r = ale_interface.act(a);
    rew += r;
    scoreCount += r;
    auto lives = ale_interface.lives();
    bool life_lost = (lives < prelives);
    go = ale_interface.game_over();
    episodeDone = go || life_lost || framesReached;
    if (episodeDone) {
      if (go || framesReached) {
        ale_interface.reset_game();
        lastEpisodeScore = scoreCount;
        lastEpisodeSteps = stepCount;
        stepCount = 0;
        scoreCount = 0;
      }
      resetObs();
      break;
    }
    if (t >= frameSkip - 2) {
      auto idx = t - (frameSkip - 2);
      ale_interface.getScreenGrayscale(rawFrames.at(idx));
    }
  }
  updateObs();
  return rew;
}

auto AtariEnv::gameOver() const -> bool { return go; }

auto AtariEnv::done() const -> bool { return episodeDone; }

auto AtariEnv::maxStepReached() const -> bool { return framesReached; }

auto AtariEnv::episodeSteps() const -> int { return lastEpisodeSteps; }
auto AtariEnv::episodeScore() const -> reward_t { return lastEpisodeScore; }

void AtariEnv::softReset() {
  episodeDone = false;
  go = false;
  framesReached = false;
}

void AtariEnv::resetCounts() {
  stepCount = 0;
  scoreCount = 0;
}

void AtariEnv::getObs(std::span<pixel_t> stateBuffer) const {
  auto it = currentObsFrame;
  std::rotate_copy(obsWin.begin(), it, obsWin.end(), stateBuffer.begin());
}

void AtariEnv::getRGB(uint8_t *screenBuffer){
  std::vector<pixel_t> v(3*rawHeight*rawWidth);
  ale_interface.getScreenRGB(v);
  std::copy(v.begin(),v.end(),screenBuffer);
}

auto AtariEnv::getMemory() const -> ReplayBuffer & { return *memory; }

auto AtariEnv::getALE() -> ale::ALEInterface & { return ale_interface; }

void AtariEnv::resetObs() {
  ale_interface.getScreenGrayscale(rawFrames.at(1));
  std::fill(obsWin.begin(), obsWin.end(), 0);
  std::fill(rawFrames.at(0).begin(), rawFrames.at(0).end(), 0);
  currentObsFrame = obsWin.begin();
  updateObs();
}

void AtariEnv::updateObs() {
  // Max pool screens #3 and #4
  std::transform(rawFrames.at(0).begin(), rawFrames.at(0).end(),
                 rawFrames.at(1).begin(), rawFrames.at(0).begin(),
                 [](auto a, auto b) { return std::max(a, b); });
  cv::Mat rawView(rawHeight, rawWidth, 0, &rawFrames.at(0)[0]);
  // Rescale to 84x84 and update the current state
  auto curridx = currentObsFrame - obsWin.begin();
  cv::Mat obsView(obsWidth, obsWidth, 0, &obsWin[curridx]);
  cv::resize(rawView, obsView, obsView.size(), 0, 0, cv::INTER_AREA);
  currentObsFrame = std::next(currentObsFrame, frameSize);
  if (currentObsFrame == obsWin.end()) {
    currentObsFrame = obsWin.begin();
  }
}
} // namespace c2d
