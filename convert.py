import importlib
from keras import backend as K
import keras
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

import tensorflow as tf

PACKAGE_NAME = "quantize"

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

models = importlib.import_module("keras-retinanet.keras_retinanet.models", package=PACKAGE_NAME)

def with_activation_layers(model):
    # boxes, scores, labels = model.outputs
    out_names = ["boxes", "scores", "labels"]
    ident_outputs = []
    for out, name in zip(model.outputs, out_names):
        identity = keras.layers.Activation("linear", name="ident_{}".format(name))(out)
        ident_outputs.append(identity)

    return keras.models.Model(inputs=model.inputs, outputs=ident_outputs)


assert len(sys.argv) == 2, "Takes one argument: h5 path"
final_model = with_activation_layers(models.convert_model(models.load_model(sys.argv[1])))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


print("Output names: ", [out.op.name for out in final_model.outputs])
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in final_model.outputs])

os.makedirs('./model', exist_ok=True)
tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)