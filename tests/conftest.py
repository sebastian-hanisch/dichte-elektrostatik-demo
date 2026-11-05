"""Gemeinsame Fixtures: die Handbeispiele und zwei Landschaften, an denen jede Einheit geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dla_constants as C  # noqa: E402
import dla_landscape as L  # noqa: E402
from dla_instance import hand_instance, make_instance  # noqa: E402


@pytest.fixture
def hand():
    """Drei Plätze nebeneinander (Abstände 1, 1, 2), Fluss A–B: 5, B–C: 1, A–C: 0. Handrechnung in der README."""
    return hand_instance()


@pytest.fixture
def hand_grid():
    """Raster 2×2, Fluss A–B 3, A–C 1, B–D 2, C–D 4 (Handbeispiel aus Stück 1)."""
    return hand_instance(C.HAND_GRID_FLOWS, *C.HAND_GRID_GRID)


@pytest.fixture
def hand_gap():
    """Vier Plätze auf einer Zeile, Fluss A–C, A–D, B–C, B–D je 1: die Schranke liegt unter dem Optimum."""
    return hand_instance(C.HAND_GAP_FLOWS, *C.HAND_GAP_GRID)


@pytest.fixture(scope="session")
def land6():
    inst = make_instance("netlist", 6, 35)
    return inst, L.landscape(inst)


@pytest.fixture(scope="session")
def land8():
    inst = make_instance("uniform", 8, 35)
    return inst, L.landscape(inst)
