// main.cpp

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <SDL.h>
}

#include <iostream>

int main(int argc, char* argv[]) {
    const char* filepath = "E:/workspace/srs/trunk/doc/source.200kbps.768x320.flv";

    av_log_set_level(AV_LOG_DEBUG);
    avformat_network_init();

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, filepath, nullptr, nullptr) < 0) {
        std::cerr << "无法打开视频文件\n";
        return -1;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "无法找到流信息\n";
        return -1;
    }

    int video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        std::cerr << "未找到视频流\n";
        return -1;
    }

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "无法找到解码器\n";
        return -1;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "无法打开解码器\n";
        return -1;
    }

    // SDL 初始化
    if (SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL 初始化失败: " << SDL_GetError() << "\n";
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("FLV Player",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        codec_ctx->width, codec_ctx->height, 0);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_YV12,
        SDL_TEXTUREACCESS_STREAMING, codec_ctx->width, codec_ctx->height);

    AVFrame* frame = av_frame_alloc();
    AVFrame* yuv = av_frame_alloc();

    uint8_t* buffer = (uint8_t*)av_malloc(
        av_image_get_buffer_size(AV_PIX_FMT_YUV420P, codec_ctx->width, codec_ctx->height, 1));
    av_image_fill_arrays(yuv->data, yuv->linesize, buffer,
        AV_PIX_FMT_YUV420P, codec_ctx->width, codec_ctx->height, 1);

    SwsContext* sws_ctx = sws_getContext(
        codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
        codec_ctx->width, codec_ctx->height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVPacket packet;
    SDL_Event event;
    bool running = true;

    while (running && av_read_frame(fmt_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, &packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    sws_scale(sws_ctx,
                        frame->data, frame->linesize, 0, codec_ctx->height,
                        yuv->data, yuv->linesize);

                    SDL_UpdateYUVTexture(texture, nullptr,
                        yuv->data[0], yuv->linesize[0],
                        yuv->data[1], yuv->linesize[1],
                        yuv->data[2], yuv->linesize[2]);

                    SDL_RenderClear(renderer);
                    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                    SDL_RenderPresent(renderer);
                    SDL_Delay(33); // 简单同步

                    while (SDL_PollEvent(&event)) {
                        if (event.type == SDL_QUIT) {
                            running = false;
                        }
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }

    // 清理资源
    av_frame_free(&frame);
    av_frame_free(&yuv);
    av_free(buffer);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
