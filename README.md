# MMV_H4Cells

[![License BSD-3](https://img.shields.io/pypi/l/mmv_h4cells.svg?color=green)](https://github.com/MMV-Lab/mmv_h4cells/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mmv_h4cells.svg?color=green)](https://pypi.org/project/mmv_h4cells)
[![Python Version](https://img.shields.io/pypi/pyversions/mmv_h4cells.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/mmv_h4cells/workflows/tests/badge.svg)](https://github.com/MMV-Lab/mmv_h4cells/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/mmv_h4cells/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/mmv_h4cells)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/mmv_h4cells)](https://napari-hub.org/plugins/mmv_h4cells)

A simple plugin to help with analyzing cells in napari

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `mmv_h4cells` via [pip]:

    pip install mmv_h4cells



To install latest development version :

    pip install git+https://github.com/MMV-Lab/mmv_h4cells.git


## Documentation

This plugin was developed for semi-automatic cell analysis to determine cell sizes of individual cells.

The core functionality includes the option to include or exclude individual (cell) instances in the evaluation via the include/exclude button. After a decision has been made, the plugin automatically centers on the next instance and a new decision can be made. In addition, you can include multiple cells at the same time using the "select multiple" function. It is also possible to analyze entire ROIs at once.
 
### Get started

To get started, an instance segmentation must be loaded. This can be done simply via drag & drop. A raw image of the original data is optional, but certainly helps when deciding whether to include or exclude.
Once the layers have been loaded into napari, the plugin can be started.
If you have only interrupted the evaluation and exported the previous results, you can now import them again (the segmentation must be reloaded into napari). 

### Analysis

The analysis can be started by clicking on the "Start analysis" button. The next instance ID to be evaluated is shown next to "Start analysis at". To change the region of interest to be evaluated, a different ID can be entered there and the plugin will center on this within the next 2 decisions. Decisions are made by clicking the Include/Exclude button. If an instance is not completely recognized correctly, you can use the paint function of napari to correct this manually and then include the instance as usual using the button. The undo function can be used to undo the last decision and the "Draw own cell" button allows you to add unrecognized cells manually. This must be done cell by cell and confirmed each time using the button. The plugin does not allow other existing instances to be painted over. If this happens by mistake, a warning is displayed, oberlapping pixels are highlighted and users can either cancel via the cancel button within the warning or close the warning and correct this manually. 

When an instance is included, the respective instance is written to a segmentation layer, which can be exported using the export function. In addition, the ID, the size and the centroid are exported as a .csv file. We also export a .zarr file, which makes it possible to re-import previously exported results, for example to pause the analysis. To enable a smooth re-import, the .csv and the .zarr file must have the same name stem, so please either do not rename the files or rename them in the same way. 

For a better overview, the included/excluded/remaining instances can be viewed using the buttons at the bottom.

#### Select multiple cells

We also support the option of including several cells at once. To do so, the respective IDs must be entered at the bottom next to "Include" and then selected using the "Select multiple". This works by entering comma-separated IDs, so *1,5,100,17* would be a valid entry.

#### Select ROI

Entire ROIs can also be analyzed. To do this, simply enter the corner pixels in the "Range x" and "Range y" fields. All cells > the threshold are included; if, for example, cells that lie exactly at the edge of the ROI and are partially cut off are to be excluded, a corresponding threshold must be set.

Note: Exported ROIs cannot be re-imported.

### Hotkeys

- `k` - Include
- `g` - Exclude
- `j` - Change visibility of all label layers for better inspection
- `h` - Undo

### Don'ts

This is a tool for analyzing cells. However, we do not catch every possible error and in order for the tool to run stable, it is important to avoid some operations:

- Do not create new layers during the analysis.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"mmv_h4cells" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/MMV-Lab/mmv_h4cells/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
