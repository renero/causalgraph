import matplotlib.pyplot as plt
import pytest

from causalexplain.common import plot


def test_setup_plot_updates_rcparams():
    # Use non-TeX settings to avoid LaTeX dependency in CI.
    plot.setup_plot(usetex=False, font_family="monospace", font_size=12, dpi=50)
    assert plt.rcParams["text.usetex"] is False
    assert plt.rcParams["font.family"][0] == "monospace"
    assert plt.rcParams["figure.dpi"] == 50


def test_add_grid_with_locations_and_without_lines():
    fig, ax = plt.subplots()
    plot.add_grid(ax, lines=False, locations=(1, 2, 3, 4))
    # With lines disabled, grid is not toggled; locator spacing still applied
    locator = ax.xaxis.get_minor_locator()
    assert locator.__class__.__name__ == "MultipleLocator"
    plt.close(fig)


@pytest.mark.parametrize("n,expected", [(2, "AB"), (3, "ABC"), (11, "ABCD;EFGH;IJK.")])
def test_get_subplot_mosaic_layout_known(n, expected):
    assert plot._get_subplot_mosaic_layout(n) == expected


def test_get_subplot_mosaic_layout_too_large():
    with pytest.raises(ValueError):
        plot._get_subplot_mosaic_layout(12)
