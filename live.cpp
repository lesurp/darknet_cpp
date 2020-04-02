#define OPENCV
#define GPU
#define CUDNN

#include "assert.h"
#include "darknet.h"
//#include "image.h"

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
    char **demo_names;
    image **demo_alphabet;
    int demo_classes;

    network *net;
    image buff[3];
    image buff_letter[3];
    int buff_index = 0;
    void *cap;
    float fps = 0;
    float demo_thresh = 0;
    float demo_hier = .5;
    int running = 0;

    int demo_frame = 3;
    int demo_index = 0;
    float **predictions;
    float *avg;
    int demo_done = 0;
    int demo_total = 0;
    double demo_time;

    void detect_in_thread();
    void fetch_in_thread();
    void display_in_thread();
    void remember_network(network *net);
    detection *avg_predictions(network *net, int *nboxes);

    void yolo_really();

    Detector(char *cfgfile,
             char *weightfile,
             float thresh,
             int cam_index,
             const char *filename,
             char **names,
             int classes,
             int delay,
             char *prefix,
             int avg_frames,
             float hier,
             int w,
             int h,
             int frames,
             int fullscreen)
    {
        // demo_frame = avg_frames;
        image **alphabet = load_alphabet();
        demo_names = names;
        demo_alphabet = alphabet;
        demo_classes = classes;
        demo_thresh = thresh;
        demo_hier = hier;
        printf("Demo\n");
        net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);

        srand(2222222);

        int i;
        demo_total = size_network(net);
        predictions = reinterpret_cast<float **>(calloc(demo_frame, sizeof(float *)));
        for (i = 0; i < demo_frame; ++i)
        {
            predictions[i] = reinterpret_cast<float *>(calloc(demo_total, sizeof(float)));
        }
        avg = reinterpret_cast<float *>(calloc(demo_total, sizeof(float)));

        if (filename)
        {
            printf("video file: %s\n", filename);
            cap = open_video_stream(filename, 0, 0, 0, 0);
        }
        else
        {
            cap = open_video_stream(0, cam_index, w, h, frames);
        }

        if (!cap)
            error("Couldn't connect to webcam.\n");

        buff[0] = get_image_from_stream(cap);
        buff[1] = copy_image(buff[0]);
        buff[2] = copy_image(buff[0]);
        buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
        buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
        buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

        if (!prefix)
        {
            make_window("Demo", 1352, 1013, fullscreen);
        }

        demo_time = what_time_is_it_now();
    }

    void run_once()
    {
        int count = 0;
        while (!demo_done)
        {
            buff_index = (buff_index + 1) % 3;
            fps = 1. / (what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            yolo_really();
            ++count;
        }
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
            memcpy(predictions[demo_index] + count,
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
    for (j = 0; j < demo_frame; ++j)
    {
        axpy_cpu(demo_total, 1. / demo_frame, predictions[j], 1, avg, 1);
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
        net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void Detector::yolo_really()
{
    fetch_in_thread();
    detect_in_thread();
    display_in_thread();
}

void Detector::display_in_thread()
{
    int c = show_image(buff[(buff_index + 1) % 3], "Demo", 1);
    if (c != -1)
        c = c % 256;
    if (c == 27)
    {
        demo_done = 1;
        return;
    }
    else if (c == 82)
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
    return;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
    if (x < 0 || x >= m.w || y < 0 || y >= m.h)
        return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if (c < 0 || c >= m.c)
        return 0;
    return get_pixel(m, x, y, c);
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

void Detector::fetch_in_thread()
{
    free_image(buff[buff_index]);
    buff[buff_index] = get_image_from_stream(cap);
    if (buff[buff_index].data == 0)
    {
        demo_done = 1;
        return;
    }
    buff_letter[buff_index] = letterbox_image(buff[buff_index], net->w, net->h);
    // letterbox_image_into(buff[buff_index], net->w, net->h, );
}

void Detector::detect_in_thread()
{
    bool running = true;
    float nms = .4;

    layer l = net->layers[net->n - 1];
    float *X = buff_letter[(buff_index + 2) % 3].data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    if (nms > 0)
        do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index + 2) % 3];
    draw_detections(
        display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1) % demo_frame;
    running = false;
}

int main(int argc, char *argv[])
{
    auto datacfg = argv[1];
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");

    auto cfg = argv[2];
    auto weights = argv[3];
    auto thresh = 0.5;
    auto cam_index = 1;
    auto filename = nullptr;
    auto **names = get_labels(name_list);
    auto classes = option_find_int(options, "classes", 20);
    auto frame_skip = 0;
    auto prefix = nullptr;
    auto avg = 3;
    auto hier_thresh = 0.5;
    auto width = 0;
    auto height = 0;
    auto fps = 0;
    auto fullscreen = 0;
    auto d = Detector(cfg,
                      weights,
                      thresh,
                      cam_index,
                      filename,
                      names,
                      classes,
                      frame_skip,
                      prefix,
                      avg,
                      hier_thresh,
                      width,
                      height,
                      fps,
                      fullscreen);

    for (;;)
        d.run_once();

    return 0;
}