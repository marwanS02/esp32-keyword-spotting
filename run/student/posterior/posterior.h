/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of the MicroKWS project.
 * See https://gitlab.lrz.de/de-tum-ei-eda-esl/ESD4ML/micro-kws for further
 * info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "esp_err.h"
#include <vector>
#include <memory>

#ifndef POSTERIOR_H
#define POSTERIOR_H

// The follwoing definitions are not available during unit tests

#ifndef CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH
#define CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH 35
#endif

#ifndef CONFIG_MICRO_KWS_POSTERIOR_TRIGGER_THRESHOLD_SINGLE
#define CONFIG_MICRO_KWS_POSTERIOR_TRIGGER_THRESHOLD_SINGLE 150
#endif

#ifndef CONFIG_MICRO_KWS_POSTERIOR_SUPPRESSION_MS
#define CONFIG_MICRO_KWS_POSTERIOR_SUPPRESSION_MS 100
#endif

#ifndef CONFIG_MICRO_KWS_NUM_CLASSES
#define CONFIG_MICRO_KWS_NUM_CLASSES 4
#endif

class PosteriorHandler {
 public:
  explicit PosteriorHandler(
      uint32_t history_length = CONFIG_MICRO_KWS_POSTERIOR_HISTORY_LENGTH,
      uint8_t trigger_threshold_single = CONFIG_MICRO_KWS_POSTERIOR_TRIGGER_THRESHOLD_SINGLE,
      uint32_t suppression_ms = CONFIG_MICRO_KWS_POSTERIOR_SUPPRESSION_MS,
      uint32_t category_count = CONFIG_MICRO_KWS_NUM_CLASSES);
  ~PosteriorHandler();

  esp_err_t Handle(uint8_t *new_posteriors, uint32_t time_ms, size_t *top_category_index,
                   bool *trigger);

 private:
  // Configuration
  uint32_t posterior_history_length_;
  uint32_t posterior_trigger_threshold_;
  uint32_t posterior_suppression_ms_;
  uint32_t posterior_category_count_;

  // Working variables

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  struct MovingAvg {
    uint32_t acc = 0;

    std::vector<uint8_t> buffer;
    size_t buffer_idx = 0;

    void emplace(uint8_t element)
    {
        acc += element;
        acc -= buffer[buffer_idx];
        buffer[buffer_idx] = element;

        buffer_idx++;
        if (buffer_idx == buffer.size())
            buffer_idx = 0;
    }
    uint32_t get_sum() const
    {
        return acc;
    }

    MovingAvg(size_t size) : buffer(size, 0) {}
  };

  std::vector<MovingAvg> avgs_;
  std::vector<uint32_t> supression_end_;

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
};

#endif  // POSTERIOR_H
