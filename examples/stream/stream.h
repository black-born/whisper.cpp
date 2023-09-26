//
// Created by jeremy on 31/08/2023.
//

#ifndef WHISPERCPPSTREAMING_STREAM_H
#define WHISPERCPPSTREAMING_STREAM_H

#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <string>
#include <deque>
#include <queue>
#include <jni.h>

std::string to_timestamp(int64_t t);
extern std::queue<std::string> sentenceQueue;
extern std::mutex queueMutex;
extern std::condition_variable conditionVar;

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, std::vector<std::string> argv, const whisper_params & params);

bool whisper_params_parse(int argc, std::vector<std::string> argv, whisper_params & params);

void whisper_print_usage(int /*argc*/, std::vector<std::string> argv, const whisper_params & params);

extern "C" {
JNIEXPORT int JNICALL Java_com_example_whispercppstreaming_CircularBuffer_processCircularBuffer(JNIEnv *env, jobject circularBufferInstance, int argc, jobjectArray parameters, jobject directBuffer);
}

#endif //WHISPERCPPSTREAMING_STREAM_H
