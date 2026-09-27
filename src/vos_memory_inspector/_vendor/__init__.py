"""Third-party code vendored without modification.

``davis2017_metrics.py``
    ``davis2017/metrics.py`` from https://github.com/davisvideochallenge/davis2017-evaluation
    at commit ``ac7c43fca936f9722837b7fbd337d284ba37004b`` (byte-identical,
    SHA-256 ``a71bfb6d2da563ebf50251bda9876cf0b4b6193842543b0a3a286190529e26be``).
    License: BSD 3-Clause, Copyright (c) 2020, DAVIS: Densely Annotated VIdeo
    Segmentation; full text in ``LICENSE.davis2017-evaluation``. Note: the
    upstream ``setup.cfg`` metadata says "GPL v3" while the repository LICENSE
    file is BSD 3-Clause; this copy follows the LICENSE file.

    Why vendored: the upstream package requires the GUI ``opencv-python`` build
    and unpinned pandas/scikit-learn/scipy/networkx/tqdm, which conflicts with the
    pinned ``opencv-python-headless`` of the ``eval`` extra. The module imports
    ``cv2`` (and ``skimage`` inside ``f_measure``), so it loads only when the
    ``eval`` extra is installed.
"""
