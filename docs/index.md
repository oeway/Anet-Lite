
# Deep learning with ImJoy

We provide a Python library powered by [**ImJoy**](https://imjoy.io/docs/#/) to
perform Deep Learning with U net architecture.

We provide different analysis workflows (you can also select them from the banners).
These analysis workflows are provided by additional ImJoy plugins providing
processing steps.

* [**ANNA-PALM**](annapalm-overview.md): Accelerating Single Molecule Localization Microscopy with Deep Learning.
* [**Segmentation**](segmentation-overview.md): segmentation of different cellular structures such as cell membrane, cytoplasm, or nuclei.

## ImJoy
[**ImJoy**](https://imjoy.io/docs/#/) is image processing platform with an easy to use interface powered by a Python engine running in the background. ImJoy plays a central role in most analysis workflows.

![ImJoyScreenshot](/img/imjoy-screenshot.png)

We provide links to install the different ImJoy plugins in dedicated ImJoy workspaces. Once installed, ImJoy remembers the workspaces and plugins and you simply have to open the web app and select the appropriate workspace [https://imjoy.io/#/app](https://imjoy.io/#/app)

If you press on the installation link, the ImJoy web app will open and display a dialog asking if you want to install the specified plugin. To confirm, press the `install` button.

![ImJoyScreenshot](/img/imjoy-installplugin.png)

Plugins require the **ImJoy Plugin Engine**, to perform computations in
Python. You will need to **install** it only once, but **launch** it each time
you work with ImJoy. For more information for how to install and use the pluging engine, please consult the [ImJoy documentation](https://imjoy.io/docs/#/user-manual?id=python-engine).
