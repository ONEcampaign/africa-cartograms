from lxml import etree
from svg.path import parse_path, Move, Line
from typing import Optional, NamedTuple
import geojson
import matplotlib.pyplot as plt
from shapely.geometry import shape
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

SVG_NS = "http://www.w3.org/2000/svg"


class AffineMatrix(NamedTuple):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


def extract_transform(transform_str: str) -> AffineMatrix:
    """
    Parses an SVG transform string of the form 'matrix(a,b,c,d,e,f)' into its float components.

    Args:
        transform_str: The transform attribute string.

    Returns:
        A tuple containing the matrix values (a, b, c, d, e, f).

    Raises:
        ValueError: If the transform string is not in the expected format.
    """
    if transform_str.startswith("matrix"):
        values = transform_str.strip("matrix()").split(",")
        a, b, c, d, e, f = map(float, values)
        return AffineMatrix(a, b, c, d, e, f)
    raise ValueError(f"Unsupported transform: {transform_str}")


def apply_transform_to_points(
    points: list[tuple[float, float]], matrix: AffineMatrix
) -> list[tuple[float, float]]:
    """
    Applies a 2D affine transform to a list of (x, y) points.

    Args:
        points: A list of (x, y) coordinate tuples.
        matrix: An AffineMatrix with values (a, b, c, d, e, f).

    Returns:
        A list of transformed (x, y) points.
    """
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in points]


def extract_path_polygon(
    path_data: str, n_samples: int = 20
) -> list[tuple[float, float]]:
    """
    Converts an SVG path string into a list of (x, y) points by sampling its segments.

    This function approximates curves using line interpolation.

    Args:
        path_data: The 'd' attribute from an SVG path.
        n_samples: Number of interpolation points per curve segment.

    Returns:
        A list of (x, y) coordinate tuples.
    """
    path = parse_path(path_data)
    points: list[tuple[float, float]] = []

    for segment in path:
        if isinstance(segment, (Line, Move)):
            points.append((segment.start.real, segment.start.imag))
        else:
            # For curves (CubicBezier, QuadraticBezier, Arc), sample n intermediate points
            for t in [i / n_samples for i in range(n_samples)]:
                pt = segment.point(t)
                points.append((pt.real, pt.imag))

    # Add final segment end if not already included
    if path and points and (path[-1].end.real, path[-1].end.imag) != points[-1]:
        points.append((path[-1].end.real, path[-1].end.imag))

    return points


def extract_path_elements(root: etree._Element) -> list[etree._Element]:
    """
    Extracts all <path> elements from an SVG root element that have a 'd' attribute.

    Args:
        root: The root SVG element parsed from the SVG file.

    Returns:
        A list of SVG <path> elements.
    """
    return root.xpath(".//svg:path[@d]", namespaces={"svg": SVG_NS})


def collect_all_transformed_points(
    paths: list[etree._Element],
) -> list[tuple[float, float]]:
    """
    Collects all points from the paths after applying their respective transforms.

    This is primarily used to compute the Y-axis flipping baseline.

    Args:
        paths: A list of SVG <path> elements.

    Returns:
        A list of all transformed (x, y) points.
    """
    points = []
    for path in paths:
        d_attr = path.get("d")
        transform = path.get("transform")
        matrix = extract_transform(transform)
        points = extract_path_polygon(d_attr)
        transformed = apply_transform_to_points(points, matrix)
        points.extend(transformed)
    return points


def compute_y_flip_baseline(points: list[tuple[float, float]]) -> float:
    """
    Computes the maximum Y value among all transformed points to be used as a baseline
    for vertical flipping.

    Args:
        points: A list of (x, y) points.

    Returns:
        The maximum Y coordinate value.
    """
    return max(y for _, y in points) if points else 0.0


def build_feature_from_path(
    path_el: etree._Element, y_max: float
) -> Optional[geojson.Feature]:
    """
    Constructs a GeoJSON Feature from an individual SVG <path> element.

    It applies the transform matrix, flips the Y coordinates, and reads the <title> as the name.

    Args:
        path_el: An SVG <path> element.
        y_max: The Y-axis value used for flipping the vertical orientation.

    Returns:
        A GeoJSON Feature with polygon geometry, or None if the path has no 'd' attribute.
    """
    d_attr = path_el.get("d")
    if not d_attr:
        return None

    transform = path_el.get("transform")
    matrix = extract_transform(transform)
    hex_points = extract_path_polygon(d_attr)
    coords = apply_transform_to_points(hex_points, matrix)
    coords_flipped = [(x, y_max - y) for x, y in coords]

    # Ensure the polygon is closed
    if coords_flipped[0] != coords_flipped[-1]:
        coords_flipped.append(coords_flipped[0])

    iso3 = path_el.get("id")
    title = path_el.find("svg:title", namespaces={"svg": SVG_NS})
    name = title.text.strip() if title is not None else None

    return geojson.Feature(
        geometry=geojson.Polygon([coords_flipped]),
        properties={"iso3": iso3, "name": name},
    )


def parse_svg_to_geojson(svg_file: str) -> geojson.FeatureCollection:
    """
    Parses an SVG file with transformed hexagonal <path> elements into a GeoJSON FeatureCollection.

    Each path is transformed individually, flipped vertically based on the global Y max,
    and converted into a GeoJSON polygon. Path names are read from nested <title> elements.

    Args:
        svg_file: Path to the SVG file.

    Returns:
        A GeoJSON FeatureCollection containing the transformed and labeled polygons.
    """
    tree = etree.parse(svg_file)
    root = tree.getroot()

    paths = extract_path_elements(root)
    transformed_points = collect_all_transformed_points(paths)
    y_max = compute_y_flip_baseline(transformed_points)

    features = [
        feature
        for path_el in paths
        if (feature := build_feature_from_path(path_el, y_max)) is not None
    ]

    return geojson.FeatureCollection(features)


def plot_geojson(
    filepath: str, label_property: str = "iso3", figsize: tuple[int, int] = (10, 10)
):
    """
    Plots a GeoJSON file using matplotlib, labeling each polygon with a specified property.

    Args:
        filepath: Path to the GeoJSON file.
        label_property: Feature property to use as a label on each polygon.
        figsize: Size of the matplotlib figure.
    """
    with open(filepath) as f:
        data = geojson.load(f)

    fig, ax = plt.subplots(figsize=figsize)
    patches = []
    labels = []

    for feature in data["features"]:
        geom = shape(feature["geometry"])
        if geom.geom_type == "Polygon":
            patch = MplPolygon(list(geom.exterior.coords), closed=True)
            patches.append(patch)
            labels.append(
                (
                    geom.centroid.x,
                    geom.centroid.y,
                    feature["properties"].get(label_property, ""),
                )
            )

    pc = PatchCollection(
        patches, edgecolor="black", facecolor="lightgray", linewidths=1
    )
    ax.add_collection(pc)

    for x, y, txt in labels:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    input_file = "./hexmap/africa_hexmap.svg"
    output_file = "./hexmap/africa_hexmap.geojson"

    fc = parse_svg_to_geojson(input_file)
    with open(output_file, "w") as f:
        geojson.dump(fc, f, indent=2)

    plot_geojson(output_file)
