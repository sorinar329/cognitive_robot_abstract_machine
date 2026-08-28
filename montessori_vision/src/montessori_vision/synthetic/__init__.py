"""Rendering a montessori board so a detector has something to learn from.

The renderer knows where it placed every piece and hole, so labels come for free; randomising the
viewpoint, the lighting and the materials is what makes a detector trained on renders survive the
step to a real camera.

.. note::
   :mod:`montessori_vision.synthetic.scene` and :mod:`montessori_vision.synthetic.generate` import
   bpy and are deliberately not imported here, so the package stays usable without the ``blender``
   extra installed.
"""

from montessori_vision.synthetic.projection import CameraProjection
from montessori_vision.synthetic.randomization import RandomizationRanges, SampledScene

__all__ = ["CameraProjection", "RandomizationRanges", "SampledScene"]
