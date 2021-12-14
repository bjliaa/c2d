#ifndef REPLAYBUFFER_HPP
#define REPLAYBUFFER_HPP
#include "common.hpp"
#include <random>

namespace c2d {
using CompressedState = std::vector<char>;
constexpr int compressBound = (stateSize + (stateSize / 255) + 16);
// Holds experiences in the form of contiguous memory blocks, one for each
// item type. Memory is partly optimized by saving only end states.
class ReplayBuffer {
public:
  explicit ReplayBuffer(int memSize);
  // Stores an experience (s,ar,es,d) by copying an end state view
  void add(action_t a, reward_t r, std::span<pixel_t> es, bool d);
  // Samples uniformly with size decided by the BatchView
  void sample(BatchView batchseg);
  // Samples a random integer between 0 and n
  [[nodiscard]] auto sampleInteger(int n) -> int;

private:
  std::default_random_engine rng;
  std::uniform_int_distribution<int> dist;
  int maxSize;
  int currentMemoryPosition;
  int currentSize;
  std::vector<action_t> amem;
  std::vector<float> rmem;
  std::vector<CompressedState> esmem;
  std::vector<float> dmem;
  auto sampleIndices(int n) -> std::vector<int>;
  void addScalars(action_t a, reward_t r, bool d);
  static void compress(std::span<pixel_t> state, CompressedState &cstate);
  static void decompress(const CompressedState &cstate,
                         std::span<pixel_t> state);
};
} // namespace c2d
#endif // REPLAYBUFFER_HPP
