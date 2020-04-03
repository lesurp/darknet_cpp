#include "darknet/detector.hpp"
#include <fstream>

// TODO: get rid of asserts when with have contracts (or whatever is replacing them)

namespace
{
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}
void set_pixel(image m, int x, int y, int c, float val)
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
image mat_to_image(cv::Mat const &src)
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
std::vector<std::string> load_labels(std::string const &labels_path)
{
    std::ifstream f(labels_path);
    std::vector<std::string> labels;
    for (std::string label; std::getline(f, label);)
    {
        labels.emplace_back(std::move(label));
    }
    labels.shrink_to_fit();
    return labels;
}

} // namespace

namespace darknet
{

struct Detector::Image : public image
{
    Image(image &&i) : image(i) {}
    Image &operator=(image &&i)
    {
        free_image(*this);
        new (this) Image(std::move(i));
        return *this;
    }
};
void Detector::free_image_wrap(Image *i) { free_image(*static_cast<image *>(i)); }

Detector::Detector(char *cfgfile,
                   char *weightfile,
                   char *labelfile,
                   float thresh,
                   int /* delay */,
                   int avg_frames,
                   float hier,
                   cv::Mat const &init_f)
    : net(load_network(cfgfile, weightfile, 0)), detection_threshold_(thresh),
      hier_threshold_(hier), avg_frames_(avg_frames),
      classes_(net->layers[net->n - 1].classes), number_layers_(size_network(net.get())),
      labels_(load_labels(labelfile)), buff_(new Image(mat_to_image(init_f))),
      buff_letter_(new Image(letterbox_image(*buff_, net->w, net->h))),
      avg_(std::make_unique<float[]>(number_layers_))
{
    set_batch_network(net.get(), 1);

    predictions_.resize(avg_frames_);
    for (auto &pred : predictions_)
    {
        pred = std::make_unique<float[]>(number_layers_);
    }
}

Detector Detector::from_data_cfg(char *cfgfile,
                                 char *weightfile,
                                 char *datacfg,
                                 float thresh,
                                 int delay,
                                 int avg_frames,
                                 float hier,
                                 cv::Mat const &init_f)
{
    list *options = read_data_cfg(datacfg);
    char *labelfile =
        option_find_str(options, (char *)"names", (char *)"data/names.list");
    return Detector(
        cfgfile, weightfile, labelfile, thresh, delay, avg_frames, hier, init_f);
}

Detections Detector::run(cv::Mat const &image)
{
    *buff_ = mat_to_image(image);
    assert(buff_->data != 0);
    letterbox_image_into(*buff_, net->w, net->h, *buff_letter_);

    float nms = .4;

    layer l = net->layers[net->n - 1];
    float *X = buff_letter_->data;
    network_predict(net.get(), X);

    remember_network(net.get());
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net.get(), &nboxes);

    if (nms > 0)
        do_nms_obj(dets, nboxes, l.classes, nms);

    avg_index = (avg_index + 1) % avg_frames_;
    return Detections{dets, nboxes};
}

void Detector::remember_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(predictions_[avg_index].get() + count,
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
    fill_cpu(number_layers_, 0, avg_.get(), 1);
    for (j = 0; j < avg_frames_; ++j)
    {
        axpy_cpu(
            number_layers_, 1. / avg_frames_, predictions_[j].get(), 1, avg_.get(), 1);
    }
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(l.output, avg_.get() + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(
        net, buff_->w, buff_->h, detection_threshold_, hier_threshold_, 0, 1, nboxes);
    return dets;
}
} // namespace darknet
