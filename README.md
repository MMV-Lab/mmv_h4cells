# ISAS-Cell-Analyzer

[![License BSD-3](https://img.shields.io/pypi/l/ISAS-Cell-Analyzer.svg?color=green)](https://github.com/MMV-Lab/ISAS-Cell-Analyzer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ISAS-Cell-Analyzer.svg?color=green)](https://pypi.org/project/ISAS-Cell-Analyzer)
[![Python Version](https://img.shields.io/pypi/pyversions/ISAS-Cell-Analyzer.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/ISAS-Cell-Analyzer/workflows/tests/badge.svg)](https://github.com/MMV-Lab/ISAS-Cell-Analyzer/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/ISAS-Cell-Analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/ISAS-Cell-Analyzer)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/ISAS-Cell-Analyzer)](https://napari-hub.org/plugins/ISAS-Cell-Analyzer)

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

You can install `ISAS-Cell-Analyzer` via [pip]:

    pip install ISAS-Cell-Analyzer



To install latest development version :

    pip install git+https://github.com/MMV-Lab/ISAS-Cell-Analyzer.git


## Documentation

This plugin was developed for semi-automatic cell analysis.

The core functionality includes the option to include or exclude individual (cell) instances in the evaluation via the include/exclude button. After a decision has been made, the system automatically centers on the next instance and a new decision can be made.

### Get started

To get started, an instance segmentation must be loaded. This can be done simply via drag & drop. A raw image of the original data is optional, but certainly helps when deciding whether to include or exclude.

Once the layers have been loaded into napari, the plugin can be started.

If you have only interrupted the evaluation and exported the previous results, you can now import them again (the segmentation must be reloaded into napari). 

### Analysis

The analysis can be started by clicking on the "Start analysis" button. The next instance ID to be evaluated is shown next to "Start analysis at". To change the ROI to be evaluated, a different ID can be entered there and the plugin will center on this within the next 2 decisions. Each decision can be made using the include/exclude button. Each decision can be made using the include/exclude button. Each decision can be made using the Include/Exclude button. If an instance is not completely recognized correctly, you can use the paint function of napari to correct this manually and then include the instance as usual using the button. The undo function can be used to undo the last decision and the "Draw own cell" button allows you to add unrecognized cells manually. This must be done cell by cell and confirmed each time using the button. The plugin does not allow other existing instances to be painted over. If this happens by mistake, a notification is displayed and users must correct this manually.

When an instance is included, the respective instance is written to a segmentation layer, which can be exported using the export function. In addition, the ID, the size and the centroid are exported as a .csv file.


show included, show excluded, show remaining, visible, hotkeys, tooltips, resolution, no layer adjustments

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"ISAS-Cell-Analyzer" is free and open source software

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

[file an issue]: https://github.com/MMV-Lab/ISAS-Cell-Analyzer/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
