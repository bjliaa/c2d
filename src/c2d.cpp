#include "atarienv.hpp"
#include "common.hpp"
// For data exchange with single environment Python wrappers
extern "C" {
c2d::AtariEnv *newEnv() { return new c2d::AtariEnv(); }
void delEnv(c2d::AtariEnv *env) { delete env; }
void initEnv(c2d::AtariEnv *env, const char *game, int memSize) {
  env->initialize(game, memSize);
}
void resetEnv(c2d::AtariEnv *env) { env->softReset(); }
void hardResetEnv(c2d::AtariEnv *env) {
  auto &ale = env->getALE();
  ale.reset_game();
  env->resetObs();
  env->softReset();
  env->resetCounts();
}
void getObsEnv(c2d::AtariEnv *env, uint8_t *stateBuffer) {
  env->getObs({stateBuffer, c2d::stateSize});
}
int getRawWidthEnv(c2d::AtariEnv *env) {
  return env->rawWidth;
}
int getRawHeightEnv(c2d::AtariEnv *env) {
  return env->rawHeight;
}
void getRGBEnv(c2d::AtariEnv *env, uint8_t *screenBuffer) {
  env->getRGB(screenBuffer);
}
int actionLengthEnv(c2d::AtariEnv *env) { return env->getActionLength(); }
int episodeStepsEnv(c2d::AtariEnv *env) { return env->episodeSteps(); }
int episodeScoreEnv(c2d::AtariEnv *env) { return env->episodeScore(); }
int actEnv(c2d::AtariEnv *env, uint8_t action, bool evalmode) {
  return env->act(action, evalmode);
}
bool doneEnv(c2d::AtariEnv *env) { return env->done(); }
bool gameOverEnv(c2d::AtariEnv *env) { return env->gameOver(); }
bool maxStepReachedEnv(c2d::AtariEnv *env) { return env->maxStepReached(); }
void prefetchBatchEnv(c2d::AtariEnv *env, uint8_t *sbuff, uint8_t *abuff,
                      float *rbuff, uint8_t *esbuff, float *dbuff, int pos,
                      int count) {
  c2d::BatchView batch{{sbuff, c2d::batchSizeOne * c2d::stateSize},
                          {abuff, c2d::batchSizeOne},
                          {rbuff, c2d::batchSizeOne},
                          {esbuff, c2d::batchSizeOne * c2d::stateSize},
                          {dbuff, c2d::batchSizeOne}};
  env->getMemory().sample(batch.subspan(pos, count));
}
}
