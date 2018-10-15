# Anet-Lite


## Generated data during training
During training, the Anet ImJoy plugin stores the model (`.hdf5` files) together with necessary events to call Tensorflow in a folder `__model__` within the data folder **TODO - currently in the workspace**. Each training has a dedicated user-defined name,
allowing to store results of different trainings.

The example below shows the `__model__` folder containg the results of two trainings
(`test1`, `test2`). The file `test1 x_model__.hdf5` containes the model, the
folder `test1 xlogs` the events file for TensorBoard.

```
__model__
│   test1 x_model__.hdf5
│   test2 x_model__.hdf5
│
└───test1 xlogs
│   │   events.out.tfevents.1539359258.HOSTURL
|
└───test2 xlogs
│   │   events.out.tfevents.1539359258.HOSTURL
│
```

### Using Tensorboard
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
