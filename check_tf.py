import tensorflow as tf

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

# We use our "load_graph" function
graph = load_graph("./model/tf_model.pb")

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)     # <--- printing the operations snapshot below
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/input_1:0')
# Tensor("ident_boxes/Identity:0", shape=(?, 300, 4), dtype=float32)
# Tensor("ident_scores/Identity:0", shape=(?, 300), dtype=float32)
# Tensor("ident_labels/Identity:0", shape=(?, 300), dtype=int32)
boxes_out = graph.get_tensor_by_name("prefix/ident_boxes/Identity:0")
scores_out = graph.get_tensor_by_name("prefix/ident_scores/Identity:0")
labels_out = graph.get_tensor_by_name("prefix/ident_labels/Identity:0")
# classification_out = graph.get_tensor_by_name("prefix/classification/concat:0")
# regression_out = graph.get_tensor_by_name("prefix/regression/concat:0")
# y = graph.get_tensor_by_name('prefix/prediction_restore:0')

# We launch a Session
# with tf.Session(graph=graph) as sess:

    # test_features = [[0.377745556,0.009904444,0.063231111,0.009904444,0.003734444,0.002914444,0.008633333,0.000471111,0.009642222,0.05406,0.050163333,7e-05,0.006528889,0.000314444,0.00649,0.043956667,0.016816667,0.001644444,0.016906667,0.00204,0.027342222,0.13864]]
        # compute the predicted output for test_x
    # pred_y = sess.run( y, feed_dict={x: test_features} )