#ifndef DETECTOR_HPP_YDSL4IM7
#define DETECTOR_HPP_YDSL4IM7

// NOTE: I had some very crtp based solution to avoid including this,
// but client code needs to access the detection objects
// We could just "manually" wrap this, but this sucks
// The cleanest method would be to patch darknet, but then all the crtp solutions would be
// unneeded :)
#define OPENCV
#define GPU
#define CUDNN
#include <darknet.h>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace darknet
{
namespace details
{
struct Image
{
    int w;
    int h;
    int c;
    std::unique_ptr<float[]> data;

    Image() : w(0), h(0), c(0), data(nullptr) {}

    Image(int w, int h, int c)
        : w(w), h(h), c(c), data(std::make_unique<float[]>(w * h * c))
    {
    }
};
} // namespace details

class Detections
{
  public:
    using iterator = detection *;
    Detections() : d_(nullptr), n_(0) {}
    Detections(detection *d, int n) : d_(d), n_(n) {}
    ~Detections()
    {
        if (d_)
        {
            free_detections(d_, n_);
        }
    }
    Detections(Detections const &) = delete;
    Detections &operator=(Detections const &) = delete;
    Detections(Detections &&d)
    {
        d_ = d.d_;
        n_ = d.n_;
        d.d_ = nullptr;
    }

    Detections &operator=(Detections &&d)
    {
        d_ = d.d_;
        n_ = d.n_;
        d.d_ = nullptr;
        return *this;
    }

    iterator begin() const { return d_; }
    iterator end() const { return d_ + n_; }

  private:
    detection *d_;
    int n_;
};

class Detector
{
  public:
    Detector(char *cfgfile,
             char *weightfile,
             char *labelfile,
             float thresh,
             int delay,
             int avg_frames,
             float hier);

    static Detector from_data_cfg(char *cfgfile,
                                  char *weightfile,
                                  char *datacfg,
                                  float thresh,
                                  int delay,
                                  int avg_frames,
                                  float hier);

    float detection_threshold() const { return detection_threshold_; }
    float &detection_threshold() { return detection_threshold_; }

    float hier_threshold() const { return hier_threshold_; }
    float &hier_threshold() { return hier_threshold_; }

    std::vector<std::string> labels() const { return labels_; }
    int classes() const { return classes_; }

    Detections run(cv::Mat const &image);

  private:
    template <auto fn> using Deleter = std::integral_constant<decltype(fn), fn>;
    std::unique_ptr<network, Deleter<free_network>> net;

    float detection_threshold_;
    // TODO: not sure about the name (wth is this?)
    float hier_threshold_;
    int avg_frames_;
    int classes_;
    int number_layers_;

    std::vector<std::string> labels_;

    details::Image buff_;
    details::Image buff_letter_;

    int avg_index = 0;
    std::vector<std::unique_ptr<float[]>> predictions_;
    std::unique_ptr<float[]> avg_;

    void remember_network(network *net);
    detection *avg_predictions(network *net, int *nboxes);
};
} // namespace darknet

#endif /* end of include guard: DETECTOR_HPP_YDSL4IM7 */
