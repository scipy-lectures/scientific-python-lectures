import numpy as np

from traits.api import HasTraits, Array, Instance, Float, Property
from traits.api import DelegatesTo
from traitsui.api import View, Item, Group
from chaco.chaco_plot_editor import ChacoPlotItem

from reservoir import Reservoir


class ReservoirEvolution(HasTraits):
    reservoir = Instance(Reservoir)

    name = DelegatesTo('reservoir')

    inflows = Array(dtype=np.float64, shape=(None))
    releass = Array(dtype=np.float64, shape=(None))

    initial_stock = Float
    stock = Property(depends_on='inflows, releases, initial_stock')

    month = Property(depends_on='stock')

    ### Traits view ##########################################################
    traits_view = View(
        Item('name'),
        Group(
            ChacoPlotItem('month', 'stock', show_label=False),
        ),
        width = 500,
        resizable = True
    )

    ### Traits properties ####################################################
    def _get_stock(self):
        """
        fixme: should handle cases where we go over the max storage
        """
        return  self.initial_stock + (self.inflows - self.releases).cumsum()

    def _get_month(self):
        return np.arange(self.stock.size)

if __name__ == '__main__':
    reservoir = Reservoir(
                            name = 'Project A',
                            max_storage = 30,
                            max_release = 100.0,
                            head = 60,
                            efficiency = 0.8
                        )

    initial_stock = 10.
    inflows_ts = np.array([6., 6, 4, 4, 1, 2, 0, 0, 3, 1, 5, 3])
    releases_ts = np.array([4., 5, 3, 5, 3, 5, 5, 3, 2, 1, 3, 3])

    view = ReservoirEvolution(
                                reservoir = reservoir,
                                inflows = inflows_ts,
                                releases = releases_ts
                            )
    view.configure_traits()
