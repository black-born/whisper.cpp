// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//

#include "common.h"
#include "whisper.h"
#include "stream.h"
#include "whisper.h"
#include "ggml.h"
#include <jni.h>

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
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <atomic>

#define UNUSED(x) (void)(x)
#define TAG "Transcribing"

#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,     TAG, __VA_ARGS__)


static size_t asset_read(void *ctx, void *output, size_t read_size) {
    return AAsset_read((AAsset *) ctx, output, read_size);
}

static bool asset_is_eof(void *ctx) {
    return AAsset_getRemainingLength64((AAsset *) ctx) <= 0;
}

static void asset_close(void *ctx) {
    AAsset_close((AAsset *) ctx);
}

static struct whisper_context *whisper_init_from_asset(
        JNIEnv *env,
        jobject assetManager,
        const char *asset_path
) {
    LOGV("Loading model from asset '%s'\n", asset_path);
    AAssetManager *asset_manager = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(asset_manager, asset_path, AASSET_MODE_STREAMING);
    if (!asset) {
        LOGW("Failed to open '%s'\n", asset_path);
        return NULL;
    }

    whisper_model_loader loader = {
            .context = asset,
            .read = &asset_read,
            .eof = &asset_is_eof,
            .close = &asset_close
    };

    return whisper_init(&loader);
}

std::queue<std::string> sentenceQueue;
std::mutex queueMutex;
std::condition_variable conditionVar;
std::atomic<bool>* stopLoopFlag = new std::atomic<bool>(false);

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    //snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);
    LOGV("%s %d %02d:%02d.%03d", buf, sizeof(buf), (int) min, (int) sec, (int) msec); // Constructor not found
    return std::string(buf);
}

bool whisper_params_parse(int argc, std::vector<std::string> argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                 arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                 arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                 arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"  || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"  || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")  { params.tinydiarize   = true; }

        else {
            //fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            LOGE("error: unknown argument: %s", arg.c_str()); // Constructor not found
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

void whisper_print_usage(int /*argc*/, std::vector<std::string> argv, const whisper_params & params) {
    LOGV("\n");
    LOGV("usage: %s [options]\n", argv[0].c_str());
    LOGV("\n");
    LOGV("options:\n");
    LOGV("  -h,       --help          [default] show this help message and exit\n");
    LOGV("  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    LOGV("            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    LOGV("            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    LOGV("            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    LOGV("  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    LOGV("  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    LOGV("  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    LOGV("  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    LOGV("  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    LOGV("  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    LOGV("  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    LOGV("  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    LOGV("  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    LOGV("  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    LOGV("  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    LOGV("  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    LOGV("  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    LOGV("  -tdrz,     --tinydiarize  [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    LOGV("\n");
}

void byteArrayToFloatVector(JNIEnv *env, jbyteArray byteArray, std::vector<float>& floatVector, int n_samples_30s) {
    jsize length = env->GetArrayLength(byteArray);
    jbyte* byteArrayElements = env->GetByteArrayElements(byteArray, NULL);

    for (int i = 0; i < length; i += sizeof(float)) {
        float floatValue;
        // Assuming little-endian byte order
        memcpy(&floatValue, &byteArrayElements[i], sizeof(float));
        floatVector.push_back(floatValue);
    }
    LOGV("vector size: %zu", floatVector.size());
    env->ReleaseByteArrayElements(byteArray, byteArrayElements, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_whispercppstreaming_HomeFragment_stopLoop(JNIEnv *env, jobject /* this */) {
    stopLoopFlag->store(true); // Stop the loop using the pointer
}

extern "C" JNIEXPORT int JNICALL Java_com_example_whispercppstreaming_CircularBuffer_processCircularBuffer(JNIEnv *env, jobject circularBufferInstance, int argc, jobjectArray parameters, jobject assetManager, jstring assetPathStr) {

    whisper_params params;

    std::vector<std::string> argv;

    jsize length = env->GetArrayLength(parameters);

    for (jsize i = 0; i < length; i++) {
        auto element = static_cast<jstring>(env->GetObjectArrayElement(parameters, i));
        if (element != nullptr) {
            const char* str = env->GetStringUTFChars(element, nullptr);
            if (str != nullptr) {
                argv.emplace_back(str);
                env->ReleaseStringUTFChars(element, str);
            }
            env->DeleteLocalRef(element);
        }
    }

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    jclass circularBuffer = env->FindClass("com/example/whispercppstreaming/CircularBuffer");
    jmethodID get = env->GetMethodID(circularBuffer, "get", "(I)[B");
    if (get == nullptr) {
        LOGE("%s", "Circular buffer get method not found"); // Constructor not found
        return 1;
    }
    jmethodID clear = env->GetMethodID(circularBuffer, "clear", "()V");
    if (clear == nullptr) {
        LOGE("%s", "Circular buffer clear method not found"); // Constructor not found
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio
    /*audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }*/

    // audio.resume();

    // whisper init

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        LOGE("error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    //LOGV"%s%s", "Current path ", std::__fs::filesystem::current_path().c_str());
    //for (const auto & entry : std::__fs::filesystem::directory_iterator(std::__fs::filesystem::current_path().c_str()))
        //LOGV"%s%s", "Available directory: ", entry.path().c_str());
    //LOGV"%s%s", "trying to find model here ", params.model.c_str());

    //struct whisper_context * ctx = whisper_init_from_file(params.model.c_str());

    struct whisper_context *ctx = NULL;
    const char *asset_path_chars = (*env).GetStringUTFChars(assetPathStr, JNI_FALSE);
    ctx = whisper_init_from_asset(env, assetManager, asset_path_chars);
    (*env).ReleaseStringUTFChars(assetPathStr, asset_path_chars);

    if (ctx == nullptr) {LOGV("%s", "failed to find model here");}
    else {LOGV("%s", "Model load Successfully");}
    std::vector<float> pcmf32_old;

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        LOGV("\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
            }
        }
        LOGV("processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            LOGV("n_new_line = %d, no_context = %d\n", n_new_line, params.no_context);
        } else {
            LOGV("using VAD, will transcribe on speech activity\n");
        }

        LOGV("\n");
    }

    int n_iter = 0;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            LOGE("failed to open output file '%s'!\n", params.fname_out.c_str());
            return 1;
        }
    }

    LOGV("[Start speaking]");
    //fflush(stdout);
    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;


    // main audio loop
    while (!stopLoopFlag->load()) {

        std::vector<float> pcmf32;
        std::vector<float> pcmf32_new;

        if (stopLoopFlag->load()) {
            __android_log_print(ANDROID_LOG_VERBOSE, "Stopping", "Transcribing successfully stopped");
            whisper_free(ctx);
            return 0;
        }

        // process new audio

        if (!use_vad) {
            while (true) {

                if (stopLoopFlag->load()) {
                    __android_log_print(ANDROID_LOG_VERBOSE, "Stopping", "Transcribing successfully stopped");
                    whisper_free(ctx);
                    return 0;
                }
                jbyteArray audio = (jbyteArray)env->CallObjectMethod(circularBufferInstance, get, params.step_ms);
                byteArrayToFloatVector(env, audio, pcmf32_new, n_samples_30s);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    LOGW("WARNING: buffer full, cannot process audio fast enough, dropping audio ...");
                    LOGW("%d audio size, %d sample step", pcmf32_new.size(), 2*n_samples_step);
                    env->CallVoidMethod(circularBufferInstance, clear);
                    //audio.clear();
                    std::vector<float>().swap(pcmf32_new);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    env->CallVoidMethod(circularBufferInstance, clear);
                    LOGV("Collected enough data, let's start !");
                    //audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            LOGV("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            jbyteArray audio = (jbyteArray)env->CallObjectMethod(circularBufferInstance, get, 2000);
            byteArrayToFloatVector(env, audio, pcmf32_new, n_samples_30s);
            // audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                jbyteArray audio = (jbyteArray)env->CallObjectMethod(circularBufferInstance, get, params.step_ms);
                byteArrayToFloatVector(env, audio, pcmf32, n_samples_30s);
                // audio.get(params.length_ms, pcmf32);
                LOGV("Threshold overcome!");
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                LOGW("Threshold not overcome, speak louder");
                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;

            wparams.audio_ctx        = params.audio_ctx;
            wparams.speed_up         = params.speed_up;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                LOGE("%s: failed to process audio\n", argv[0].c_str());
                return 6;
            }

            // print result;
            {
                if (!use_vad) {
                    LOGV("[2K");

                    // print long empty line to clear the previous line
                    LOGV("%s", std::string(100, ' ').c_str());

                    LOGV("[2K");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    LOGV("\n");
                    LOGV("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    LOGV("\n");
                }

                    const int n_segments = whisper_full_n_segments(ctx);
                LOGV("n_segment: %d", n_segments);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        LOGV("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0) + " --> " + to_timestamp(t1) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        LOGV("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad){
                    LOGV("\n");
                    LOGV("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                // Get the last sentence and send it to other thread
                const int n_segments = whisper_full_n_segments(ctx);
                if (0 < n_segments) {
                    std::string text = (const char *) whisper_full_get_segment_text(ctx,
                                                                                    n_segments - 1);
                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        sentenceQueue.push(text);
                    }
                    // Notify the condition variable
                    conditionVar.notify_one();
                }
                else {
                    LOGV("Number of segment is 0, Nothing translatable inside the record");
                }

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }

    __android_log_print(ANDROID_LOG_VERBOSE, "Stopping", "Transcribing successfully stopped");
    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
