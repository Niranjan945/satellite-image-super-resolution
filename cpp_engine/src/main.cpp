// Stage 6: C++ Inference Engine (core version)
// Loads the exported ONNX model and runs Real-ESRGAN super-resolution
// entirely in C++ - no Python required. Reports inference time so it
// can be compared against the Python timings from Stage 3.
//
// Usage: sr_engine.exe <input_image_path> <output_image_path>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

static const wchar_t* MODEL_PATH = L"models/onnx/RealESRGAN_x4plus.onnx";

std::vector<float> preprocess(const cv::Mat& bgr_image) {
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    int h = float_image.rows;
    int w = float_image.cols;
    std::vector<float> chw_data(3 * h * w);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(chw_data.data() + c * h * w, channels[c].ptr<float>(), h * w * sizeof(float));
    }
    return chw_data;
}

cv::Mat postprocess(const float* data, int height, int width) {
    cv::Mat channels_r(height, width, CV_32FC1, (void*)(data + 0 * height * width));
    cv::Mat channels_g(height, width, CV_32FC1, (void*)(data + 1 * height * width));
    cv::Mat channels_b(height, width, CV_32FC1, (void*)(data + 2 * height * width));

    std::vector<cv::Mat> channels = {channels_r, channels_g, channels_b};
    cv::Mat rgb_float;
    cv::merge(channels, rgb_float);

    cv::Mat rgb_uint8;
    rgb_float = cv::max(0.0f, cv::min(1.0f, rgb_float));
    rgb_float.convertTo(rgb_uint8, CV_8UC3, 255.0);

    cv::Mat bgr_uint8;
    cv::cvtColor(rgb_uint8, bgr_uint8, cv::COLOR_RGB2BGR);
    return bgr_uint8;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    cv::Mat input_image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: could not read input image: " << input_path << "\n";
        return 1;
    }
    int in_h = input_image.rows;
    int in_w = input_image.cols;
    std::cout << "Loaded input image: " << in_w << "x" << in_h << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sr_engine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    Ort::Session session(env, MODEL_PATH, session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    std::vector<float> input_data = preprocess(input_image);
    std::array<int64_t, 4> input_shape = {1, 3, in_h, in_w};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    std::cout << "Running Real-ESRGAN inference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    double inference_seconds = std::chrono::duration<double>(t1 - t0).count();

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int out_h = static_cast<int>(output_shape[2]);
    int out_w = static_cast<int>(output_shape[3]);

    cv::Mat output_image = postprocess(output_data, out_h, out_w);
    cv::imwrite(output_path, output_image);

    std::cout << "Output image: " << out_w << "x" << out_h << "\n";
    std::cout << "Inference time: " << inference_seconds << "s\n";
    std::cout << "Saved to: " << output_path << "\n";
    return 0;
}