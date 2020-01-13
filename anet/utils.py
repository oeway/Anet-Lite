
def export_model_to_js(model, path, remove_input_size=True):
    import tensorflowjs as tfjs
    input_shape_bk = None
    if remove_input_size:
        # make input image size adaptive
        input_layer = model.get_layer(index=0)
        input_shape_bk = input_layer.batch_input_shape
        input_layer.batch_input_shape = (None, None, None, input_shape_bk[3])
    tfjs.converters.save_keras_model(model, path)
    # recover shape
    if remove_input_size and input_shape_bk and input_layer:
        input_layer.batch_input_shape = input_shape_bk

