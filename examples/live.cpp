#include "assert.h"
#include "darknet/detector.hpp"
#include <fmt/printf.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

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
    auto d =
        darknet::Detector(datacfg, cfg, weights, thresh, frame_skip, avg, hier_thresh, f);

    auto start = what_time_is_it_now();
    for (;;)
    {
        cap >> f;
        auto detections = d.run(f);

        auto end = what_time_is_it_now();
        auto fps = 1. / (end - start);
        start = end;
        fmt::printf("\nFPS:%.1f\n", fps);
        fmt::printf("Objects:\n\n");
        draw_detections(f, detections, d.detection_threshold(), d.labels(), d.classes());
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
