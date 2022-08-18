#!/usr/bin/env python3
# cython: language_level=3
# coding=utf-8 
import sys
import cv2
import math
import time
import pyds
import ctypes
import platform
import numpy as np
import configparser
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib


from utils.utils import make_element, is_aarch64, create_source_bin, bus_call
from utils.utils import get_outshape, decode, postprocess
from utils.display import add_obj_meta

MUX_OUTPUT_WIDTH = 1280
MUX_OUTPUT_HEIGHT = 720
INFER_SHAPE = (1,3,640,640)


start_time = time.time()
start_time2 = 0
vid_writer = None
save_video = False
aspect_ratio = float(192 / 256)  # w/h
DEBUG = True
codec = "H264"
INPUT_STREAM = [
    # "rtsp://admin:zc62683949@172.16.240.91:554/h264/ch1/main/av_stream",
    # "file:////home/nvidia/project/lightweight-human-pose-estimation.pytorch/17人1080p.mp4"
    "file:///home/nvidia/project/yolo-pose/merge.mp4"
    ]

vid_writer = None


def tracker_sink_pad_buffer_probe(pad, info, u_data):
    t = time.time()
    global vid_writer
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if DEBUG:
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # convert python array into numy array format.
            frame_image = np.array(n_frame, copy=True, order="C")
            # covert the array into cv2 default color format
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        l_usr = frame_meta.frame_user_meta_list
        
        while l_usr is not None:
            try:
                # Casting l_obj.data to pyds.NvDsUserMeta
                user_meta = pyds.NvDsUserMeta.cast(l_usr.data)
            except StopIteration:
                break

            # get tensor output
            if (user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):  # NVDSINFER_TENSOR_OUTPUT_META
                try:
                    l_usr = l_usr.next
                except StopIteration:
                    break
                # continue

            try:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                infer_out = []
                # translate pointer to numpy array
                for num in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, num)
                    # print(f'output layer: {layer.layerName}')
                    # load float* buffer to python
                    stride = int(layer.layerName.replace('stride_',''))
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    out_shape = get_outshape(INFER_SHAPE,stride)
                    # print(out_shape)
                    infer_out.append(np.ctypeslib.as_array(ptr, shape=out_shape))
                decode_out = decode(infer_out,INFER_SHAPE[2:],strides=[8,16,32,64])
                if DEBUG:
                    pred = postprocess(decode_out,(MUX_OUTPUT_HEIGHT,MUX_OUTPUT_WIDTH),INFER_SHAPE[2:],frame_image)
                    boxes, confs, kpts = pred
                else:
                    pred = postprocess(decode_out,(MUX_OUTPUT_HEIGHT,MUX_OUTPUT_WIDTH),INFER_SHAPE[2:])
                    boxes, confs, kpts = pred
                if len(boxes)>0 and len(confs)>0 and len(kpts)>0:
                    add_obj_meta(frame_meta,batch_meta,boxes[0],confs[0])
            except StopIteration:
                break

            try:
                l_usr = l_usr.next
            except StopIteration:
                break

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
        # print(boxes )
        num_person = sum([len(box) for box in boxes])
        display_text = f'{cur_time} Person={num_person} Frames={frame_number} FPS={CurFPS:.0f} AvgFPS={AvgFPS:.1f}'
        print(display_text)
        
        start_time = time.time()
        if int(start_time2) == 0:
            start_time2 = time.time()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
        # Indicating inference is performed on given frame.
        pyds.nvds_acquire_meta_lock(batch_meta)
        frame_meta.bInferDone=True
        pyds.nvds_release_meta_lock(batch_meta)

    cost = time.time()-t
    print(f'probe function time cost {cost*1000:.2f}ms')

        

    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))

    l_frame = batch.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            obj_meta.text_params.display_text = "person{}: {:.2f}".format(obj_meta.object_id ,obj_meta.tracker_confidence)
            track_box = obj_meta.tracker_bbox_info.org_bbox_coords
            print(track_box.left,track_box.top,track_box.height,track_box.width)
            # rect_params = obj_meta.rect_params
            # rect_params.left = track_box.left
            # rect_params.top = track_box.top
            # rect_params.width = track_box.width
            # rect_params.height = track_box.height
            l_obj = l_obj.next
        l_frame = l_frame.next
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

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_element("nvstreammux", "Stream-muxer-left")
    streammux.set_property("width", MUX_OUTPUT_WIDTH)
    streammux.set_property("height", MUX_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000)
    streammux.set_property("live-source", 1)  # rtsp
    pipeline.add(streammux)

    number_src_left = len(INPUT_STREAM)
    for i in range(number_src_left):
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


    tracker = make_element("nvtracker", "tracker")
    
    # Use nvdsanalytics to perform analytics on object
    nvdsanalytics = make_element("nvdsanalytics", "nvdsanalytics")
    nvdsanalytics.set_property("config-file", "configs/config_nvdsanalytics.txt")

    nvvidconv = make_element("nvvideoconvert", "convertor")

    nvvidconv_postosd = make_element("nvvideoconvert", "convertor_postosd")

    nvvidconv1 = make_element("nvvideoconvert", "convertor_pre")

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = make_element("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    # Create OSD to draw on the converted RGBA buffer
    nvosd = make_element("nvdsosd", "onscreendisplay")
    
    # Create a caps filter
    caps = make_element("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    # Make the encoder
    encoder = make_element("nvv4l2h264enc", "encoder")
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("preset-level", 1)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("bufapi-version", 1)

    # Make the payload-encode video into RTP packets
    rtppay = make_element("rtph264pay", "rtppay")
        
    # Make the UDP sink
    updsink_port_num = 5401
    sink = make_element("udpsink", "udpsink")
    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)
    sink.set_property("qos", 0)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read("configs/config_tracker.txt")
    config.sections()

    for key in config["tracker"]:
        if key == "tracker-width":
            tracker_width = config.getint("tracker", key)
            tracker.set_property("tracker-width", tracker_width)
        if key == "tracker-height":
            tracker_height = config.getint("tracker", key)
            tracker.set_property("tracker-height", tracker_height)
        if key == "gpu-id":
            tracker_gpu_id = config.getint("tracker", key)
            tracker.set_property("gpu_id", tracker_gpu_id)
        if key == "ll-lib-file":
            tracker_ll_lib_file = config.get("tracker", key)
            tracker.set_property("ll-lib-file", tracker_ll_lib_file)
        if key == "ll-config-file":
            tracker_ll_config_file = config.get("tracker", key)
            tracker.set_property("ll-config-file", tracker_ll_config_file)
        if key == "enable-batch-process":
            tracker_enable_batch_process = config.getint("tracker", key)
            tracker.set_property("enable_batch_process", tracker_enable_batch_process)

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = make_element("nvinfer", "primary-inference-left")
    pgie.set_property("config-file-path", "configs/config_infer_primary.txt")


    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        #tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(filter1)
    pipeline.add(tracker)
    pipeline.add(nvdsanalytics)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)
    # Link the elements together:
    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tracker)

    # nvosd -> nvvidconv -> caps -> encoder -> rtppay -> udpsink
    tracker.link(nvdsanalytics)
    nvdsanalytics.link(nvvidconv)
    nvvidconv.link(nvosd)

    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start rtsp streaming
    rtsp_port_num = 9554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, \
        encoding-name=(string)%s, payload=96 " )' % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/review", factory)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    
    trackerpad = tracker.get_static_pad("sink")
    if not trackerpad:
        sys.stderr.write(" Unable to get sink pad of tracker \n")
    trackerpad.add_probe(Gst.PadProbeType.BUFFER, tracker_sink_pad_buffer_probe, 0)
    '''
    nvanalytics_src_pad = nvdsanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad of analytics\n")
    nvanalytics_src_pad.add_probe(
        Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0
    )
    '''
    print("Starting pipeline \n")

    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        if save_video:
            vid_writer.release()

    # cleanup
    pipeline.set_state(Gst.State.NULL)
    print("End pipeline \n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
