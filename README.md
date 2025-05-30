# africa-cartograms

This repository provides template cartograms of Africa in both SVG and GeoJSON formats.
It also includes a Python module with utility functions to convert SVG path data into valid GeoJSON geometries, 
supporting workflows for geographic data visualization and transformation.

## SVG to GeoJSON example

```python
from svg_to_geojson.utils import parse_svg_to_geojson, plot_geojson
import geojson

input_file = "../hexmap/africa_hexmap.svg"
output_file = "../hexmap/africa_hexmap.geojson"

fc = parse_svg_to_geojson(input_file)
with open(output_file, "w") as f:
    geojson.dump(fc, f, indent=2)

plot_geojson(output_file)
```

## File structure

```
.
├── hexmap/  
│   ├── africa_hexmap.geojson
│   └── africa_hexmap.svg
└── svg_to_geojson
    └── utils.py
```


