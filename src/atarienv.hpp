#ifndef ATARIENV_HPP
#define ATARIENV_HPP
#include "common.hpp"
#include "replaybuffer.hpp"
#include <ale/ale_interface.hpp>
#include <set>

namespace c2d {

// Atari Environment
class AtariEnv {
public:
  // Initializes the environment.
  void initialize(const std::string &game, int memSize);
  // Returns the size of the minimal action set.
  [[nodiscard]] auto getActionLength() const -> int;
  // Steps the environment by an action chosen in [0, <size of the minimal
  // action set>).
  [[nodiscard]] auto act(action_t action, bool evalmode = false) -> reward_t;
  auto act(action_t action, std::span<pixel_t> sbuff) -> reward_t;
  // Signals that game over was reached in ALE.
  [[nodiscard]] auto gameOver() const -> bool;
  // Signals that the episodic max step was reached.
  [[nodiscard]] auto maxStepReached() const -> bool;
  // Total score in the last episode.
  [[nodiscard]] auto episodeScore() const -> reward_t;
  // Number of steps taken in the last episode.
  [[nodiscard]] auto episodeSteps() const -> int;
  // Nulls all current count statistics.
  void resetCounts();
  // Signals that a soft reset must be called.
  [[nodiscard]] auto done() const -> bool;
  // Must be called if done().
  void softReset();
  // Writes the current observation window [4 x 84 x 84] into a stateBuffer.
  void getObs(std::span<pixel_t> stateBuffer) const;
  // Direct access to ReplayBuffer
  [[nodiscard]] auto getMemory() const -> ReplayBuffer &;
  // Direct access to ALE
  [[nodiscard]] auto getALE() -> ale::ALEInterface &;
  // Nulls and initializes the observations buffer
  void resetObs();
  // RGB dimensions
  int rawHeight;
  int rawWidth;
  // RGB screen
  void getRGB(uint8_t *screenBuffer);

private:
  std::unique_ptr<ReplayBuffer> memory;
  ale::ALEInterface ale_interface;
  ale::ActionVect actionMap;
  int numActions;
  bool episodeDone = false;
  bool go = false;
  bool framesReached = false;
  int stepCount = 0;
  int lastEpisodeSteps = 0;
  reward_t scoreCount = 0;
  reward_t lastEpisodeScore = 0;
  std::array<Frame, 2> rawFrames;
  ObsWindow obsWin;
  std::vector<pixel_t>::const_iterator currentObsFrame = obsWin.begin();
  void updateObs();
  [[nodiscard]] auto act_func(action_t action) -> reward_t;
};
} // namespace c2d
#endif // ATARIENV_HPP
