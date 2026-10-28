"""DeepFurniture scene population and ground-truth boxes for the crop-before-search run.

Contract: docs/contracts/crop-before-search-deepfurniture.md (Population, Images and coordinates,
Arms).

Pure logic over parsed annotations plus the tar streaming that feeds it. Nothing here decodes an
image. Rules fixed by the contract:

- A scene is eligible when its instances include at least one val identity and no test identity
  (split of each identity from the manifest), and it is not the source scene of any val query in
  ``query_index.json``. Its answers are its distinct val identities.
- GT boxes: all instances regardless of split, ordered by ``areaRatio`` descending; instances without
  ``areaRatio`` after all others, by ``numberID`` ascending; ties by ``numberID`` ascending. Cut to
  ``M``, then the shared padding / minimum size (``crop.boxes.build_crops``).
- Boxes are native pixel coordinates. A box that is not a finite rectangle of positive area, or lies
  entirely outside its image, raises ``SceneBoxError`` (never dropped silently).

Choices where the contract is silent (reported, not hidden):
- ``xMax``/``yMax`` are treated as edge coordinates (exclusive), like the detector's boxes.
- An identity that is not in the manifest, in any scene, stops the run: its split is unknown, so the
  scene cannot be classified.
- ``areaRatio`` that is null counts as missing.
"""

from __future__ import annotations

import gzip
import json
import math
import tarfile
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from product_retrieval.crop.boxes import TOP_M, Crop, Detection, build_crops, clip_boxes

EXPECTED_SCENES = 228
EXPECTED_ANSWERS = 285
EXPECTED_GALLERY = 2527
ANNOTATION_NAME = "annotation.json"
IMAGE_NAME = "image.jpg"


class ScenePopulationError(ValueError):
    """Raised when annotations, the manifest or the expected counts do not match the contract."""


class SceneBoxError(ValueError):
    """Raised when an annotated box is unusable (not a rectangle, or outside its image)."""


@dataclass(frozen=True)
class Instance:
    number_id: int
    box: tuple[float, float, float, float]  # (xMin, yMin, xMax, yMax), native pixels
    identity: str
    category: str
    area_ratio: float | None  # None = the annotation has no areaRatio


@dataclass(frozen=True)
class SceneAnnotation:
    scene_id: str
    instances: tuple[Instance, ...]


@dataclass(frozen=True)
class ScenePopulation:
    scenes: list[str]  # ascending scene id
    answers: dict[str, list[str]]  # scene id -> sorted distinct val identities
    excluded: dict[str, int] = field(default_factory=dict)
    excluded_query_source_scenes: list[str] = field(default_factory=list)

    @property
    def n_answers(self) -> int:
        return sum(len(a) for a in self.answers.values())


def _require_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenePopulationError(f"{what}: expected an integer, got {value!r}")
    return value


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenePopulationError(f"{what}: expected a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ScenePopulationError(f"{what}: not finite: {value!r}")
    return out


def parse_annotation(scene_id: str, obj: Any) -> SceneAnnotation:
    """Validate one ``annotation.json`` object. Anything unexpected raises (nothing is guessed)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("instances"), list):
        raise ScenePopulationError(f"scene {scene_id!r}: annotation has no 'instances' list")
    scene = obj.get("scene")
    if isinstance(scene, dict) and "sceneTaskID" in scene and scene["sceneTaskID"] != scene_id:
        raise ScenePopulationError(
            f"scene {scene_id!r}: annotation says sceneTaskID {scene['sceneTaskID']!r}"
        )
    instances: list[Instance] = []
    seen: set[int] = set()
    for raw in obj["instances"]:
        where = f"scene {scene_id!r}"
        if not isinstance(raw, dict):
            raise ScenePopulationError(f"{where}: instance is not an object")
        number_id = _require_int(raw.get("numberID"), f"{where} numberID")
        where = f"{where} instance {number_id}"
        if number_id in seen:
            raise ScenePopulationError(f"{where}: duplicate numberID")
        seen.add(number_id)
        bb = raw.get("boundingBox")
        if not isinstance(bb, dict) or any(k not in bb for k in ("xMin", "xMax", "yMin", "yMax")):
            raise ScenePopulationError(f"{where}: boundingBox needs xMin/xMax/yMin/yMax")
        box = (
            _number(bb["xMin"], f"{where} xMin"),
            _number(bb["yMin"], f"{where} yMin"),
            _number(bb["xMax"], f"{where} xMax"),
            _number(bb["yMax"], f"{where} yMax"),
        )
        identity = raw.get("identityID")
        if isinstance(identity, bool) or not isinstance(identity, int | str) or str(identity) == "":
            raise ScenePopulationError(f"{where}: identityID must be an int or a non-empty string")
        category = raw.get("categoryName")
        if not isinstance(category, str) or not category:
            raise ScenePopulationError(f"{where}: categoryName must be a non-empty string")
        area = raw.get("areaRatio")
        area_ratio = None if area is None else _number(area, f"{where} areaRatio")
        instances.append(Instance(number_id, box, str(identity), category, area_ratio))
    if not instances:
        raise ScenePopulationError(f"scene {scene_id!r}: no instances")
    return SceneAnnotation(scene_id, tuple(instances))


def _iter_scene_members(
    archive: Path, wanted: Mapping[str, Collection[str] | None]
) -> Iterator[tuple[str, str, bytes]]:
    """Stream ``(scene_id, file_name, bytes)`` for ``<scene>/<file_name>`` members.

    ``wanted`` maps a file name to the scene ids it is needed for (``None`` = every scene). Other
    members are skipped without being read into memory.
    """
    try:
        with gzip.open(archive, "rb") as gz, tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2 or parts[-1] not in wanted:
                    continue
                scene_id, name = parts[-2], parts[-1]
                scenes = wanted[name]
                if scenes is not None and scene_id not in scenes:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    raise ScenePopulationError(f"{archive}: unreadable member {member.name!r}")
                data = f.read()
                if len(data) != member.size or not data:
                    raise ScenePopulationError(f"{archive}: member {member.name!r} is empty or truncated")
                yield scene_id, name, data
            # tarfile treats a truncated stream as a clean end; draining the gzip layer makes a
            # cut-off archive raise EOFError instead of passing silently.
            while gz.read(1024 * 1024):
                pass
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ScenePopulationError(f"{archive}: cannot read archive: {exc}") from exc


def read_annotations(archive: str | Path) -> dict[str, SceneAnnotation]:
    """Parse every scene's ``annotation.json`` from the scene archive (images are not read)."""
    archive = Path(archive)
    out: dict[str, SceneAnnotation] = {}
    for scene_id, _name, data in _iter_scene_members(archive, {ANNOTATION_NAME: None}):
        if scene_id in out:
            raise ScenePopulationError(f"{archive}: duplicate annotation for scene {scene_id!r}")
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ScenePopulationError(f"scene {scene_id!r}: annotation is not valid JSON: {exc}") from exc
        out[scene_id] = parse_annotation(scene_id, obj)
    if not out:
        raise ScenePopulationError(f"{archive}: no {ANNOTATION_NAME} members found")
    return out


def read_scene_images(archive: str | Path, scene_ids: Collection[str]) -> dict[str, bytes]:
    """Raw ``image.jpg`` bytes of the given scenes (kept in memory; nothing is written to disk)."""
    archive = Path(archive)
    wanted = set(scene_ids)
    out: dict[str, bytes] = {}
    for scene_id, _name, data in _iter_scene_members(archive, {IMAGE_NAME: wanted}):
        if scene_id in out:
            raise ScenePopulationError(f"{archive}: duplicate image for scene {scene_id!r}")
        out[scene_id] = data
    missing = sorted(wanted - set(out))
    if missing:
        raise ScenePopulationError(
            f"{archive}: no {IMAGE_NAME} for {len(missing)} scenes, e.g. {missing[:3]}"
        )
    return out


def val_query_source_scenes(query_index: Mapping[str, Any], split_of: Mapping[str, str]) -> set[str]:
    """Source scenes of val queries: ``<identityID>_<n>_<sceneTaskID>.jpg`` with a val identity."""
    queries = query_index.get("queries") if isinstance(query_index, Mapping) else None
    if not isinstance(queries, list) or not queries:
        raise ScenePopulationError("query_index.json has no 'queries' list")
    scenes: set[str] = set()
    for q in queries:
        name = q.get("query_name") if isinstance(q, dict) else None
        stem = name[: -len(".jpg")] if isinstance(name, str) and name.endswith(".jpg") else None
        parts = stem.split("_") if stem else []
        if len(parts) != 3 or not all(parts):
            raise ScenePopulationError(f"query name {name!r} is not <identityID>_<n>_<sceneTaskID>.jpg")
        identity, _n, scene_id = parts
        if identity not in split_of:
            raise ScenePopulationError(f"query {name!r}: identity {identity!r} is not in the manifest")
        if split_of[identity] == "val":
            scenes.add(scene_id)
    return scenes


def select_population(
    annotations: Mapping[str, SceneAnnotation],
    split_of: Mapping[str, str],
    query_source_scenes: Collection[str],
    expected_scenes: int = EXPECTED_SCENES,
    expected_answers: int = EXPECTED_ANSWERS,
) -> ScenePopulation:
    """Apply the contract's eligibility rules and assert the expected counts."""
    unknown = sorted({i.identity for a in annotations.values() for i in a.instances} - set(split_of))
    if unknown:
        raise ScenePopulationError(
            f"{len(unknown)} annotated identities are not in the manifest (split unknown), e.g. {unknown[:5]}"
        )
    query_sources = set(query_source_scenes)
    excluded = {"no_val_identity": 0, "test_identity": 0, "val_query_source": 0}
    excluded_query_scenes: list[str] = []
    scenes: list[str] = []
    answers: dict[str, list[str]] = {}
    for scene_id in sorted(annotations):
        splits = {split_of[i.identity] for i in annotations[scene_id].instances}
        if "test" in splits:
            excluded["test_identity"] += 1
        elif "val" not in splits:
            excluded["no_val_identity"] += 1
        elif scene_id in query_sources:
            excluded["val_query_source"] += 1
            excluded_query_scenes.append(scene_id)
        else:
            scenes.append(scene_id)
            answers[scene_id] = sorted(
                {i.identity for i in annotations[scene_id].instances if split_of[i.identity] == "val"}
            )
    pop = ScenePopulation(scenes, answers, excluded, excluded_query_scenes)
    if len(scenes) != expected_scenes or pop.n_answers != expected_answers:
        raise ScenePopulationError(
            f"population is {len(scenes)} scenes / {pop.n_answers} answers, "
            f"contract says {expected_scenes} / {expected_answers} (excluded: {excluded})"
        )
    return pop


def select_gallery(rows: Iterable[Any], expected: int = EXPECTED_GALLERY) -> list[tuple[str, str]]:
    """Val gallery of baseline v0: ``(product_id, image sha)`` for every val row, ascending product id."""
    gallery: dict[str, str] = {}
    for row in rows:
        if row.split != "val":
            continue
        if len(row.gallery) != 1:
            raise ScenePopulationError(
                f"val product {row.product_id!r} has {len(row.gallery)} gallery images, not 1"
            )
        if row.product_id in gallery:
            raise ScenePopulationError(f"duplicate val product {row.product_id!r} in the manifest")
        gallery[row.product_id] = row.gallery[0]
    if len(gallery) != expected:
        raise ScenePopulationError(f"val gallery has {len(gallery)} identities, contract says {expected}")
    return sorted(gallery.items())


def gt_order(instances: Iterable[Instance]) -> list[Instance]:
    """``areaRatio`` descending; missing ``areaRatio`` after all others; ties by ``numberID`` ascending."""
    return sorted(
        instances,
        key=lambda i: (
            i.area_ratio is None,
            -(i.area_ratio if i.area_ratio is not None else 0.0),
            i.number_id,
        ),
    )


def clipped_boxes(
    scene: SceneAnnotation, width: int, height: int
) -> dict[int, tuple[float, float, float, float]]:
    """Clip every instance's native box to the image; raise ``SceneBoxError`` if one is unusable."""
    if width < 1 or height < 1:
        raise SceneBoxError(f"scene {scene.scene_id!r}: image size must be positive, got {width}x{height}")
    out: dict[int, tuple[float, float, float, float]] = {}
    for inst in scene.instances:
        x1, y1, x2, y2 = inst.box
        if not (x2 > x1 and y2 > y1):
            raise SceneBoxError(
                f"scene {scene.scene_id!r} instance {inst.number_id}: box {inst.box} has no positive area"
            )
        cx1, cy1, cx2, cy2 = clip_boxes(np.array([inst.box]), width, height)[0]
        if not (cx2 > cx1 and cy2 > cy1):
            raise SceneBoxError(
                f"scene {scene.scene_id!r} instance {inst.number_id}: box {inst.box} "
                f"is outside the {width}x{height} image"
            )
        out[inst.number_id] = (float(cx1), float(cy1), float(cx2), float(cy2))
    return out


@dataclass(frozen=True)
class GtBoxes:
    crops: list[Crop]  # crop.index = numberID, in GT order, after padding and the minimum size rule
    first_m: list[Instance]  # the first M instances in GT order (before padding / minimum size)


def gt_crops(scene: SceneAnnotation, width: int, height: int, m: int = TOP_M) -> GtBoxes:
    """GT box arm: validate all boxes, order, cut to ``m``, then the shared padding and minimum size."""
    boxes = clipped_boxes(scene, width, height)  # validates every instance, not only the first m
    first_m = gt_order(scene.instances)[:m]
    detections = [
        Detection(
            index=i.number_id, score=i.area_ratio if i.area_ratio is not None else 0.0, box=boxes[i.number_id]
        )
        for i in first_m
    ]
    return GtBoxes(crops=build_crops(detections, width, height), first_m=first_m)


def answers_without_box(
    scene: SceneAnnotation, answers: Collection[str], kept_numbers: Collection[int]
) -> list[str]:
    """Answers (identities) none of whose instances has its numberID in ``kept_numbers``."""
    kept = set(kept_numbers)
    have = {i.identity for i in scene.instances if i.number_id in kept}
    return sorted(set(answers) - have)


def answer_categories(scene: SceneAnnotation, answers: Collection[str]) -> list[tuple[str, str]]:
    """Distinct ``(identity, category)`` pairs of the scene's answers (one per category an identity has)."""
    wanted = set(answers)
    pairs: dict[str, set[str]] = defaultdict(set)
    for i in scene.instances:
        if i.identity in wanted:
            pairs[i.identity].add(i.category)
    return sorted((ident, cat) for ident, cats in pairs.items() for cat in cats)
