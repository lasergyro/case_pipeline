from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from traceback import print_exc
from typing import Protocol, TypeVar, cast

# import beartype
# import beartype.door
import numpy as np
import numpy.typing as npt
from ovito.data import DataCollection, PropertyContainer
from ovito.modifiers import PythonScriptModifier
from ovito.pipeline import Pipeline, PipelineSourceInterface, PythonScriptSource
from traits.api import String, observe

NDint = npt.NDArray[np.integer]
NDfloat = npt.NDArray[np.floating]
NDbool = npt.NDArray[np.bool_]


class CasePath(Protocol):
    @property
    def path(self) -> Path:
        ...


CasePathType = TypeVar("CasePathType", bound=CasePath)
T = TypeVar("T")


def cast_assert(x, T2: type[T]) -> T:
    # beartype.door.die_if_unbearable(x, T2)
    return cast(T, x)


def notnone(x: T | None) -> T:
    assert x is not None
    return x


def update_container(key: str, data: DataCollection, data2: DataCollection):
    h = getattr(data, key, None)
    h2 = getattr(data2, key + "_", None)

    def up(a: PropertyContainer, b: PropertyContainer):
        if b.count != a.count:
            b.count = a.count
        for k, v in a.items():
            if k not in b:
                if 0 != a.standard_property_type_id(k):
                    prop = b.create_property(k, data=np.array(v))
                else:
                    prop = b.create_property(
                        k, components=v.component_count, data=np.array(v)
                    )
                    prop.component_names = v.component_names
                # prop.types_ = v.types
            b[k + "_"][:] = v[:]

    if (h2 is None) and (h is None):
        pass
    elif (h2 is None) and (h is not None):
        # create, update
        if key == "particles":
            h2 = data2.create_particles()
        else:
            raise ValueError
        up(h, h2)

    elif (h2 is not None) and (h is None):
        # delete
        del data2[key]
    elif (h2 is not None) and (h is not None):
        # update
        up(h, h2)


def copy_to(data: DataCollection, data2: DataCollection):
    if data2.cell is None:
        data2.create_cell(np.array(data.cell), pbc=data.cell.pbc)
    data2.cell_[:] = data.cell[:]
    update_container("particles", data, data2)


@dataclass
class CaseCache:
    path: Path
    num_frames: int = field(init=False, repr=False, default=0)
    pipeline: Pipeline = field(init=False, repr=False)
    _init_done: bool = field(init=False, repr=False, default=False)
    _data_path: Path | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._frame = np.empty(0, dtype=int)
        self.pipeline = self._make_pipeline()
        self.refresh(force=True)

    def __len__(self):
        return self.num_frames

    def get_data_path(self):
        return self.path / "in.data"

    def _make_pipeline(self):
        from ovito.io import import_file

        # Data import:
        pipeline = import_file(self.get_data_path(), atom_style="atomic")
        from ovito.modifiers import LoadTrajectoryModifier

        # Load trajectory:
        mod = LoadTrajectoryModifier()
        pipeline.modifiers.append(mod)
        mod.source.load(str(self.path / "traj.atom"), multiple_frames=True)
        self.num_frames = mod.source.num_frames
        return pipeline

    def _init_data(self, frame: int, data: DataCollection):
        copy_to(self.pipeline.compute(frame), data)

    def refresh_traj(self) -> bool:
        return False

    def _update_data(
        self,
        frame: int,
        data: DataCollection,
    ):
        copy_to(self.pipeline.compute(frame), data)

    def refresh(self, force: bool = False):
        changed = force
        if self._data_path is None:
            self._data_path = self.get_data_path()
            changed = changed or self._data_path is not None
        changed = changed or self.refresh_traj()
        return changed

    def _update(self, frame: int, data: DataCollection):
        changed = False
        if data.cell is None:
            if self._init_done:
                changed = self.refresh()
            self._init_done = False
        if not self._init_done:
            self._init_data(0, data)
            self._init_done = True
        self._update_data(frame, data)
        return changed

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


_wref = dict()


class CasePipelineSource(PipelineSourceInterface):
    path = String("")

    def cache(self):
        if id(self) not in _wref:
            _wref[id(self)] = {}
            weakref.finalize(self, partial(_wref.pop, id(self), None))
        cache = _wref[id(self)]
        return cache

    def case_cache(self) -> CaseCache:
        return self.cache()["cp"]

    @observe("path")
    def update_path(self, event):
        try:
            cache = self.cache()
            cp = cache.get("cp", None)
            if cp is not None:
                cp.close()
                del cache["cp"]
            cache["cp"] = CaseCache(
                path=Path(str(self.path)),
            )
        except:
            if os.environ.get("OVITO_GUI", None):
                print_exc()
            else:
                raise
        self.notify_trajectory_length_changed()

    def refresh(self):
        if self.case_cache().refresh():
            self.notify_trajectory_length_changed()
            return True
        return False

    def compute_trajectory_length(self, **kwargs) -> int:
        cache = self.cache()
        cp: CaseCache | None = cache.get("cp", None)
        if cp is None:
            return 0
        else:
            return len(cp)

    def create(self, data: DataCollection, *, frame: int, **kwargs):
        cache = self.cache()

        cp: CaseCache | None = cache.get("cp", None)
        if cp is not None:
            if frame >= len(cp):  # assume animation
                frame = len(cp) - 1
            if cp._update(frame, data):
                self.notify_trajectory_length_changed()

    def close(self):
        cp = self.cache().get("cp", None)
        if cp is not None:
            cp.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
