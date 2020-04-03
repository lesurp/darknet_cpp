#include "assert.h"
#include "darknet/detector.hpp"
#include <chrono>
#include <fmt/printf.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

void draw_detections(cv::Mat &f,
                     darknet::Detections const &detections,
                     float detection_threshold,
                     std::vector<std::string> labels)
{
    for (auto det : detections)
    {
        int best_class = -1;
        float best_detection = detection_threshold;
        for (int j = 0; j < det.classes; ++j)
        {
            if (det.prob[j] > best_detection)
            {
                best_class = j;
                best_detection = det.prob[j];
                break;
            }
        }
        if (best_class < 0)
        {
            continue;
        }

        float red = 255.0;
        float green = 255.0;
        float blue = 0.0;
        box b = det.bbox;

        int left = (b.x - b.w / 2.) * f.cols;
        int right = (b.x + b.w / 2.) * f.cols;
        int top = (b.y - b.h / 2.) * f.rows;
        int bot = (b.y + b.h / 2.) * f.rows;

        if (left < 0)
            left = 0;
        if (right > f.cols - 1)
            right = f.cols - 1;
        if (top < 0)
            top = 0;
        if (bot > f.rows - 1)
            bot = f.rows - 1;

        cv::rectangle(f,
                      cv::Point(left, top),
                      cv::Point(right, bot),
                      cv::Scalar(red, green, blue),
                      1);
        cv::putText(f,
                    labels[best_class],
                    cv::Point(left, bot),
                    0,
                    1.0,
                    cv::Scalar_(red, green, blue));
        // TODO: what is this..?
        // it's never defined it seems
        /*
        if (det.mask)
        {
            image mask = float_to_image(14, 14, 1, dets[i].mask);
            image resized_mask = resize_image(mask, b.w * f.cols, b.h * f.rows);
            image tmask = threshold_image(resized_mask, .5);
            embed_image(tmask, im, left, top);
            free_image(mask);
            free_image(resized_mask);
            free_image(tmask);
        }
        */
    }
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        fmt::print("Not enough arguments, 5 required");
        return 1;
    }

    auto datacfg = argv[1];
    auto cfg = argv[2];
    auto weights = argv[3];
    auto cam = argv[4];
    auto thresh = 0.5;
    // auto classes = option_find_int(options, (char *)"classes", 20);
    auto frame_skip = 0;
    auto avg = 3;
    auto hier_thresh = 0.5;

    cv::VideoCapture cap(cam);
    if (!cap.isOpened())
    {
        fmt::print("Could not open camera device: {}", cam);
        return 1;
    }

    cv::Mat f;
    cap >> f;
    auto d = darknet::Detector::from_data_cfg(
        cfg, weights, datacfg, thresh, frame_skip, avg, hier_thresh, f);

    auto start = std::chrono::steady_clock::now();
    for (;;)
    {
        cap >> f;
        auto detections = d.run(f);

        auto end = std::chrono::steady_clock::now();
        auto fps = double(std::chrono::steady_clock::period::den) / (end - start).count();
        start = end;
        fmt::print("\nFPS:{}\n", fps);
        fmt::print("Objects:\n\n");
        draw_detections(f, detections, d.detection_threshold(), d.labels());
        cv::imshow("asd", f);
        char c = cv::waitKey(1);
        switch (c)
        {
        case 'q':
            return 0;
        case 'j':
            d.detection_threshold() -= 0.02;
            d.detection_threshold() = std::max(0.02f, d.detection_threshold());
            break;
        case 'k':
            d.detection_threshold() += 0.02;
            break;
            // TODO: what is this for?
        case 'h':
            d.hier_threshold() -= 0.02;
            d.detection_threshold() = std::max(0.0f, d.detection_threshold());
            break;
        case 'l':
            d.hier_threshold() += 0.02;
            break;
        }
    }

    return 0;
}
