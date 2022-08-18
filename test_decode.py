#!/usr/bin/env python3
# cython: language_level=3
# coding=utf-8
import os
import sys
import cv2
import time
import pyds
import ctypes
import platform
import numpy as np
import configparser
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # 设置cuda同步，方便debug

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

from utils.utils import make_element, is_aarch64, create_source_bin, bus_call
from utils.utils import get_total_outshape, postprocess


MUX_OUTPUT_WIDTH = 640 
MUX_OUTPUT_HEIGHT =640 
INFER_SHAPE = (1,3,640,640)
OUT_SHAPE = get_total_outshape(INFER_SHAPE)
INPUT_STREAM = ["file:///media/nvidia/SD/project/test/2in1_2.mp4"]
DEBUG = False

codec = "H264"
start_time = time.time()
start_time2 = 0
vid_writer = None
data = dict()
data['conf_thres'] = 0.1
data['iou_thres'] = 0.45


def tracker_sink_pad_buffer_probe(pad, info, u_data):
    t = time.time()
    global  data,vid_writer
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if DEBUG:
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_image = np.array(n_frame, copy=True, order="C")
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

        frame_number = frame_meta.frame_num

        if DEBUG:
            if vid_writer == None:  # new video
                vid_writer = cv2.VideoWriter(
                    "record.avi",
                    cv2.VideoWriter_fourcc("X", "V", "I", "D"),
                    25,
                    (frame_image.shape[1], frame_image.shape[0]),
                )

            vid_writer.write(frame_image.copy())

        
        global start_time, start_time2
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        CurFPS = 1 / (time.time() - start_time)
        AvgFPS = frame_number / (time.time() - start_time2)
        
        display_text = f'{cur_time} Frames={frame_number} FPS={CurFPS:.0f} AvgFPS={AvgFPS:.1f}'
        print(display_text)
        
        start_time = time.time()
        if int(start_time2) == 0:
            start_time2 = time.time()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    print(f'probe function time cost:{(time.time()-t)*1000:.2f}ms')
    return Gst.PadProbeReturn.OK


def main(args):
    # Standard GStreamer initialization
    # Since version 3.11, calling threads_init is no longer needed. See: https://wiki.gnome.org/PyGObject/Threading
    # GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    nvvidconv_postosd = make_element("nvvideoconvert", "convertor_postosd")

    nvvidconv1 = make_element("nvvideoconvert", "convertor_pre")

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = make_element("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    # Create a caps filter
    caps = make_element("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    fakesink = make_element('fakesink','sink')
    fakesink.set_property('silent', 1)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_element("nvstreammux", "Stream-muxer-left")
    streammux.set_property("width", MUX_OUTPUT_WIDTH)
    streammux.set_property("height", MUX_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 1)  # rtsp
    pipeline.add(streammux)

    number_src = len(INPUT_STREAM)
    for i in range(number_src):
        print("Creating source_bin ", i, " \n ")
        uri_name = INPUT_STREAM[i]
        print(uri_name)

        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)

        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    print("Adding elements to Pipeline \n")
    pipeline.add(nvvidconv_postosd)
    pipeline.add(nvvidconv1)
    pipeline.add(filter1)
    pipeline.add(caps)
    pipeline.add(fakesink)

    # Link the elements together:
    print("Linking elements in the Pipeline \n")
    streammux.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(fakesink)

    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgiepad = nvvidconv_postosd.get_static_pad("src")
    if not pgiepad:
        sys.stderr.write(" Unable to get sink pad of tracker \n")
    pgiepad.add_probe(Gst.PadProbeType.BUFFER, tracker_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")

    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        if vid_writer:
            vid_writer.release()
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    print("End pipeline \n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
