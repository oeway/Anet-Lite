# Anet-Lite

A generic plugin for image-to-image translation with A-net.

## Usage

### Data organization
The plugin requires a strict data organization, with dedicated folders for
**training** (`train`), **validation** (`valid`), and **testing** (`test`) with
imposed names. Each sample has its own folder, which can be named freely. Each
sample folder contains different images (input and/or target), which are named
identically among different samples.

Below shows an example for such an organization. The sample folders contain
one input channel (`channelA.png`) and a target channel (`channelB.png`). You can
also choose different channel names (e.g. `DAPI`, `C5`), but the names have to
be consistent accross all the sample folder. Within the plugin, you can then
define regular expression to indicate what the input and target channels are.
Note that the `test` folder only contains the input channel.

```
.
├─ train/
│  ├─ sample1/
│  │  ├─ channelA.png
│  │  ├─ channelB.png
│  ├─ sample2/
│  │  ├─ channelA.png
│  │  ├─ channelB.png
│  ├─ ...
├─ valid/
│  ├─ sample20/
│  │  ├─ channelA.png
│  │  ├─ channelB.png
│  ├─ ...
├─ test/
│  ├─ sample43/
│  │  ├─ channelA.png
│  ├─ sample44/
│  │  ├─ channelA.png
│  ├─ ...
.
```

### Set working directory

Once the data is organized, you can set the `set working directory` of the analysis,
i.e. the folder containing the training and validation data.

And you will be asked to specify the identifiers for your plugin.

## Training
Training requires folders `train` and `valid`, which need to be in
the specified working directory.

You can change a few parameters to dermine how the training is performed.
**TODO** add screen shots

* You can assign a name to each training instance. This name will be be used as
* a prefix when saving the model and logs.
Choose the `epochs`, `step per epoch` and `batch size` you want to train.

Then you can start training by clicking on `???`. Training progress will be
indicated in the status bar of ImJoy. Already during training several data
are generated (see next section), which can for intance be used to monitor
training progress with TensorBoard.

Once the training is done, you will get your model located in the
`working directory`, the folder will be called `__model__`. You can now use this
model to predict your test data, or you can load the model file
named `*__model__.h5`, where `*` is your specified name, if you want to initiate
another training instance with this model (aka warm start).

**TODO** currently saved elsewhere


### Generated data during training
During training, the Anet ImJoy plugin stores the model (`.hdf5` files) together with necessary events to call Tensorflow in a folder `__model__` within the data folder **TODO - currently in the workspace**. Each training has a dedicated user-defined name,
allowing to store results of different trainings.

The example below shows the `__model__` folder containg the results of two trainings
(`test1`, `test2`). The file `test1 x_model__.hdf5` containes the model, the
folder `test1 xlogs` the events file for TensorBoard.

```
.
├─ __model__/
├─ test1 x_model__.hdf5
├─ test2 x_model__.hdf5
│
├─ test1 xlogs/
│  ├─ events.out.tfevents.1539359258.HOSTURL
│
├─ test2 xlogs/
│  ├─ events.out.tfevents.1539359260.HOSTURL
.
```
### Advanced: using Tensorboard
[TensorBoard](https://www.tensorflow.org/) provides a suite of visualization tools to make it easier to understand, debug, and optimize DeepLearning programs. To use
Tensorboard, open a terminal and navigate to the folder containing the events
file, e.g. `test1 xlogs` from the example above.

Then call TensorBoard with the following command.
```
python -m tensorboard.main --logdir=.
TensorBoard 1.5.1 at LOCALHOST:6006 (Press CTRL+C to quit)
```
You will obtain the local host `LOCALHOST` address to open Tensorboard, open a browser and paste the entire address.

If you receive an error message such as `This site can’t be reached`, replace the local host address by `127.0.01`, e.g. for the example above: `http://127.0.0.1:6006`.

## Optionally: load trained models

**TODO**


## Testing/Prediction
Once you have a trained model (either from your own data or by loading an
already trained model), you can apply this model to the `test` data.

Place your files with all the input channels in a folder `test` and you will be able to run prediction on them.

By clicking `test` in the plugin menu, you will start the prediction.

Once done, the result will be saved automatically into the testing folder.
