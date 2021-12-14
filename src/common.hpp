#ifndef COMMON_HPP
#define COMMON_HPP
#include <cstdint>
#include <span>
#include <vector>

namespace c2d {

constexpr int batchSizeOne = 32;
constexpr int frameSkip = 4;
constexpr int obsStack = 4;
constexpr int obsWidth = 84;
constexpr int frameSize = obsWidth * obsWidth;
constexpr int stateSize = obsStack * frameSize;
constexpr int endStepMax = 27000;
constexpr float repeatActionProbability = 0.25;

using pixel_t = uint8_t;
using action_t = uint8_t;
using reward_t = int;
using done_t = float;
using Frame = std::vector<pixel_t>;
using FlatState = std::vector<pixel_t>;
using ObsWindow = std::vector<pixel_t>;

// Consolidates views of external batch buffers
struct BatchView {
  std::span<pixel_t> bs{};
  std::span<action_t> ba{};
  std::span<float> br{};
  std::span<pixel_t> bes{};
  std::span<float> bd{};
  [[nodiscard]] auto subspan(int pos, int count) const -> BatchView {
    return BatchView{bs.subspan(pos * stateSize, count * stateSize),
                     ba.subspan(pos, count), br.subspan(pos, count),
                     bes.subspan(pos * stateSize, count * stateSize),
                     bd.subspan(pos, count)};
  }
};

} // namespace c2d
#endif // COMMON_HPP
