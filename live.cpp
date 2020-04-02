#include <opencv2/core/types.hpp>
#define OPENCV
#define GPU
#define CUDNN

#include "assert.h"
#include "darknet.h"
#include <fmt/printf.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

void draw_detections(
    cv::Mat &f, detection *dets, int num, float thresh, char **names, int classes)
{
    for (int i = 0; i < num; ++i)
    {
        int class_ = -1;
        for (int j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > thresh)
            {
                // note that we could have other values above the threshold, but the
                // classes are ordered in decreasing order
                class_ = j;
                break;
            }
        }
        if (class_ < 0)
        {
            continue;
        }

        float red = 255.0;
        float green = 255.0;
        float blue = 0.0;
        box b = dets[i].bbox;

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
        // TODO: what is this..?
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

class Detector
{
  public:
    Detector(char *datacfgfile,
             char *cfgfile,
             char *weightfile,
             float thresh,
             int /* delay */,
             int avg_frames,
             float hier,
             cv::Mat const &init_f)
        : net(load_network(cfgfile, weightfile, 0)), detection_threshold_(thresh),
          hier_threshold_(hier), avg_frames_(avg_frames),
          classes_(net->layers[net->n - 1].classes), number_layers_(size_network(net))
    {
        list *options = read_data_cfg(datacfgfile);
        char *name_list =
            option_find_str(options, (char *)"names", (char *)"data/names.list");

        labels_ = get_labels(name_list);
        set_batch_network(net, 1);

        predictions = reinterpret_cast<float **>(calloc(avg_frames_, sizeof(float *)));
        for (int i = 0; i < avg_frames_; ++i)
        {
            predictions[i] =
                reinterpret_cast<float *>(calloc(number_layers_, sizeof(float)));
        }
        avg = reinterpret_cast<float *>(calloc(number_layers_, sizeof(float)));

        buff = mat_to_image(init_f);
        buff_letter = letterbox_image(buff, net->w, net->h);
    }

    float detection_threshold() const { return detection_threshold_; }
    float &detection_threshold() { return detection_threshold_; }

    float hier_threshold() const { return hier_threshold_; }
    float &hier_threshold() { return hier_threshold_; }

    char **labels() const { return labels_; }
    int classes() const { return classes_; }

    std::pair<detection *, int> run(cv::Mat const &image)
    {
        free_image(buff);
        buff = mat_to_image(image);
        assert(buff.data != 0);
        letterbox_image_into(buff, net->w, net->h, buff_letter);

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

        avg_index = (avg_index + 1) % avg_frames_;
        return {dets, nboxes};
    }

  private:
    network *net;
    float detection_threshold_;
    float hier_threshold_;
    int avg_frames_;
    int classes_;
    int number_layers_;

    char **labels_;

    image buff;
    image buff_letter;

    int avg_index = 0;
    float **predictions;
    float *avg;
    // TODO: not sure about the name (wth is this?)

    void remember_network(network *net);
    detection *avg_predictions(network *net, int *nboxes);

    // TODO dtor

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
    fill_cpu(number_layers_, 0, avg, 1);
    for (j = 0; j < avg_frames_; ++j)
    {
        axpy_cpu(number_layers_, 1. / avg_frames_, predictions[j], 1, avg, 1);
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
        net, buff.w, buff.h, detection_threshold_, hier_threshold_, 0, 1, nboxes);
    return dets;
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
    auto d = Detector(datacfg, cfg, weights, thresh, frame_skip, avg, hier_thresh, f);

    make_window((char *)"Demo", 1352, 1013, 0);
    auto start = what_time_is_it_now();
    for (;;)
    {
        cap >> f;
        auto [dets, nboxes] = d.run(f);

        auto end = what_time_is_it_now();
        auto fps = 1. / (end - start);
        start = end;
        fmt::printf("\nFPS:%.1f\n", fps);
        fmt::printf("Objects:\n\n");
        draw_detections(
            f, dets, nboxes, d.detection_threshold(), d.labels(), d.classes());
        // TODO: make some wrapper around this (with raii)
        free_detections(dets, nboxes);
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
