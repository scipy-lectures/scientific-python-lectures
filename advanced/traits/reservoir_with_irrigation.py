from traits.api import HasTraits, Str, Float, Range, Enum, List
from traitsui.api import View, Item

class IrrigationArea(HasTraits):
    name = Str
    surface = Float(desc='Surface [ha]')
    crop = Enum('Alfalfa', 'Wheat', 'Cotton')


class Reservoir(HasTraits):
    name = Str
    max_storage = Float(1e6, desc='Maximal storage [hm3]')
    max_release = Float(10, desc='Maximal release [m3/s]')
    head = Float(10, desc='Hydraulic head [m]')
    efficiency = Range(0, 1.)
    irrigated_areas = List(IrrigationArea)

    def energy_production(self, release):
        ''' Returns the energy production [Wh] for the given release [m3/s]
        '''
        power = 1000 * 9.81 * self.head * release * self.efficiency
        return power * 3600

    traits_view = View(
        Item('name'),
        Item('max_storage'),
        Item('max_release'),
        Item('head'),
        Item('efficiency'),
        Item('irrigated_areas'),
        resizable = True
    )

if __name__ == '__main__':
    upper_block = IrrigationArea(name='Section C', surface=2000, crop='Wheat')

    reservoir = Reservoir(
                    name='Project A',
                    max_storage=30,
                    max_release=100.0,
                    head=60,
                    efficiency=0.8,
                    irrigated_areas=[upper_block]
                )

    release = 80
    print 'Releasing {} m3/s produces {} kWh'.format(
        release, reservoir.energy_production(release)
    )
