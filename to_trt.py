import tensorflow.contrib.tensorrt as trt

import tensorflow as tf
from load_graph import load_graph_def
from tensorflow.python.framework import graph_io

GRAPH_PB_PATH = './model/tf_model.pb'

with tf.Session() as sess:
    graph_def = load_graph_def(GRAPH_PB_PATH)
    # print("load graph")
    # with tf.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    # sess.graph.as_default()
    # tf.import_graph_def(graph_def, name='')
    #    graph_nodes=[n for n in graph_def.node]
    #    names = []
    #    for t in graph_nodes:
    #       names.append(t.name)
    #    print(names)
    output_names = [
        "ident_boxes/Identity",
        "ident_scores/Identity",
        "ident_labels/Identity",
    ]
    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=5
    )

    graph_io.write_graph(trt_graph, "./model/",
        "trt_graph.pb", as_text=False)