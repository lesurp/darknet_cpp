#include <opencv2/core/types.hpp>
#define OPENCV
#define GPU
#define CUDNN

#include "assert.h"
#include "darknet.h"
#include <fmt/printf.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

void draw_detections(cv::Mat &f,
                     detection *dets,
                     int num,
                     float thresh,
                     char **names,
                     image **alphabet,
                     int classes)
{
    int i, j;

    for (i = 0; i < num; ++i)
    {
        char labelstr[4096] = {0};
        int class_ = -1;
        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                if (class_ < 0)
                {
                    strcat(labelstr, names[j]);
                    class_ = j;
                }
                else
                {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);
            }
        }
        if (class_ < 0)
        {
            continue;
        }

        float red = 1.0;
        float green = 1.0;
        float blue = 0.0;
        float rgb[3];

        // width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = dets[i].bbox;
        // printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

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
                    names[class_],
                    cv::Point(left, bot),
                    0,
                    1.0,
                    cv::Scalar_(red, green, blue));
        /*
        if (dets[i].mask)
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

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
        return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] = val;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x, y, k;
    for (k = 0; k < source.c; ++k)
    {
        for (y = 0; y < source.h; ++y)
        {
            for (x = 0; x < source.w; ++x)
            {
                float val = get_pixel(source, x, y, k);
                set_pixel(dest, dx + x, dy + y, k, val);
            }
        }
    }
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w / im.w) < ((float)h / im.h))
    {
        new_w = w;
        new_h = (im.h * w) / im.w;
    }
    else
    {
        new_h = h;
        new_w = (im.w * h) / im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
    free_image(resized);
}

int size_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            count += l.outputs;
        }
    }
    return count;
}

struct Detector
{
    float detection_threshold() const { return detection_threshold_; }
    char **labels() const { return labels_; }

    char **labels_;

    network *net;
    image buff;
    image buff_letter;
    float detection_threshold_ = 0;
    float demo_hier = .5;
    int running = 0;

    int avg_frame = 3;
    int avg_index = 0;
    float **predictions;
    float *avg;
    int demo_total = 0;

    std::pair<detection *, int> detect_in_thread();
    void remember_network(network *net);
    detection *avg_predictions(network *net, int *nboxes);

    Detector(char *cfgfile,
             char *weightfile,
             float thresh,
             char **names,
             int /* delay */,
             int avg_frames,
             float hier,
             cv::Mat const &init_f)
    {
        avg_frame = avg_frames;
        labels_ = names;
        detection_threshold_ = thresh;
        demo_hier = hier;
        printf("Demo\n");
        net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);

        int i;

        demo_total = size_network(net);
        predictions = reinterpret_cast<float **>(calloc(avg_frame, sizeof(float *)));
        for (i = 0; i < avg_frame; ++i)
        {
            predictions[i] = reinterpret_cast<float *>(calloc(demo_total, sizeof(float)));
        }
        avg = reinterpret_cast<float *>(calloc(demo_total, sizeof(float)));

        buff = mat_to_image(init_f);
        buff_letter = letterbox_image(buff, net->w, net->h);
    }

    std::pair<detection *, int> run_once(cv::Mat const &image)
    {
        free_image(buff);
        buff = mat_to_image(image);
        assert(buff.data != 0);
        letterbox_image_into(buff, net->w, net->h, buff_letter);
        return detect_in_thread();
    }

    static image mat_to_image(cv::Mat const &src)
    {
        int h = src.rows;
        int w = src.cols;
        int c = src.channels();
        image im = make_image(w, h, c);
        unsigned char *data = (unsigned char *)src.data;
        int step = w * c;

        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                for (int k = 0; k < c; ++k)
                {
                    // NOTE: we do c - k - 1, while the original code does just k,
                    // but swap the two channels manually afterwards
                    im.data[k * w * h + i * w + j] =
                        data[i * step + j * c + (c - k - 1)] / 255.;
                }
            }
        }
        return im;
    }
};

void Detector::remember_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(predictions[avg_index] + count,
                   net->layers[i].output,
                   sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *Detector::avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for (j = 0; j < avg_frame; ++j)
    {
        axpy_cpu(demo_total, 1. / avg_frame, predictions[j], 1, avg, 1);
    }
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(
        net, buff.w, buff.h, detection_threshold_, demo_hier, 0, 1, nboxes);
    return dets;
}

std::pair<detection *, int> Detector::detect_in_thread()
{
    float nms = .4;

    layer l = net->layers[net->n - 1];
    float *X = buff_letter.data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    if (nms > 0)
        do_nms_obj(dets, nboxes, l.classes, nms);

    avg_index = (avg_index + 1) % avg_frame;
    return {dets, nboxes};
}

int main(int argc, char *argv[])
{
    auto datacfg = argv[1];
    list *options = read_data_cfg(datacfg);
    char *name_list =
        option_find_str(options, (char *)"names", (char *)"data/names.list");

    auto cfg = argv[2];
    auto weights = argv[3];
    auto cam = argv[4];
    auto thresh = 0.5;
    auto **names = get_labels(name_list);
    auto classes = option_find_int(options, (char *)"classes", 20);
    auto frame_skip = 0;
    auto avg = 3;
    auto hier_thresh = 0.5;

    image **alphabet = load_alphabet();

    cv::VideoCapture cap(cam);
    if (!cap.isOpened())
    {
        fmt::print("Could not open camera device: {}", cam);
        return 1;
    }

    cv::Mat f;
    cap >> f;
    auto d = Detector(cfg, weights, thresh, names, frame_skip, avg, hier_thresh, f);

    make_window((char *)"Demo", 1352, 1013, 0);
    auto start = what_time_is_it_now();
    for (;;)
    {
        cap >> f;
        auto [dets, nboxes] = d.run_once(f);

        auto end = what_time_is_it_now();
        auto fps = 1. / (end - start);
        start = end;
        printf("\nFPS:%.1f\n", fps);
        printf("Objects:\n\n");
        draw_detections(
            f, dets, nboxes, d.detection_threshold(), d.labels(), alphabet, classes);
        free_detections(dets, nboxes);
        cv::imshow("asd", f);
        cv::waitKey(1);
        //int c = show_image(d.buff, "Demo", 1);
        //if (c != -1)
            //c = c % 256;
        /*
        if (c == 82)
        {
            demo_thresh += .02;
        }
        else if (c == 84)
        {
            demo_thresh -= .02;
            if (demo_thresh <= .02)
                demo_thresh = .02;
        }
        else if (c == 83)
        {
            demo_hier += .02;
        }
        else if (c == 81)
        {
            demo_hier -= .02;
            if (demo_hier <= .0)
                demo_hier = .0;
        }
        */
    }

    return 0;
}
