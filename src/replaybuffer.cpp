#include "replaybuffer.hpp"
#include "common.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <lz4.h>
#include <stdexcept>

namespace c2d {
ReplayBuffer::ReplayBuffer(int memSize)
    : rng(std::random_device{}()),
      maxSize(memSize), currentMemoryPosition(0), currentSize(0), amem(memSize),
      rmem(memSize), esmem(memSize), dmem(memSize) {}

void ReplayBuffer::add(action_t a, reward_t r, std::span<pixel_t> es, bool d) {
  // To save memory we only compress and store the ending state
  compress(es, esmem[currentMemoryPosition]); 
  addScalars(a, r, d);
}

void ReplayBuffer::addScalars(action_t a, reward_t r, bool d) {
  amem[currentMemoryPosition] = a;
  rmem[currentMemoryPosition] = static_cast<float>(r);
  dmem[currentMemoryPosition] = static_cast<c2d::done_t>(d);
  currentMemoryPosition =
      (currentMemoryPosition + 1 == maxSize ? 0 : currentMemoryPosition + 1);
  if (currentSize < maxSize) {
    currentSize++;
  }
}

void ReplayBuffer::sample(BatchView batchseg) {
  auto indices = sampleIndices(batchseg.ba.size());
  auto pbs = batchseg.bs;
  auto pbes = batchseg.bes;
  int count = 0;
  for (int idx : indices) {
    auto sidx = idx - 1;
    decompress(esmem[sidx], pbs.subspan(count * stateSize, stateSize));
    decompress(esmem[idx], pbes.subspan(count * stateSize, stateSize));
    ++count;
  }
  std::transform(indices.begin(), indices.end(), batchseg.ba.begin(),
                 [&](int idx) { return amem[idx]; });
  std::transform(indices.begin(), indices.end(), batchseg.br.begin(),
                 [&](int idx) { return rmem[idx]; });
  std::transform(indices.begin(), indices.end(), batchseg.bd.begin(),
                 [&](int idx) { return dmem[idx]; });
}

auto ReplayBuffer::sampleIndices(int n) -> std::vector<int> {
  dist.param(
      std::uniform_int_distribution<int>::param_type(1, currentSize - 1));
  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(rng); });
  return vec;
}

auto ReplayBuffer::sampleInteger(int n) -> int {
  dist.param(std::uniform_int_distribution<int>::param_type(1, n));
  return dist(rng);
}

void ReplayBuffer::compress(std::span<pixel_t> state, CompressedState &cstate) {
  CompressedState buffer(compressBound);
  auto compressionBytes =
      LZ4_compress_default(reinterpret_cast<const char *>(state.data()),
                           buffer.data(), stateSize, compressBound);
  if (compressionBytes < 1) {
    std::cerr << "Compression failed (lz4 bytes < 1).";
    throw std::runtime_error("Compression error.");
  }
  cstate.resize(compressionBytes);
  cstate.shrink_to_fit();
  std::copy_n(buffer.begin(), compressionBytes, cstate.begin());
}

void ReplayBuffer::decompress(const CompressedState &cstate,
                              std::span<pixel_t> state) {
  auto decompressionBytes =
      LZ4_decompress_safe(cstate.data(), reinterpret_cast<char *>(state.data()),
                          cstate.size(), stateSize);
  if (decompressionBytes < 1) {
    std::cerr << "Decompression failed (lz4 bytes < 1).";
    throw std::runtime_error("Decompression error.");
  }
}

} // namespace c2d
