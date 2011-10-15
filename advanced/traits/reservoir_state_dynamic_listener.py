from reservoir import Reservoir
from reservoir_state_property import ReservoirState

def wake_up_watchman_if_spillage(new_value):
    if new_value > 0:
        print 'Wake up watchman! Spilling {} hm3'.format(new_value)

if __name__ == '__main__':
    projectA = Reservoir(
                        name = 'Project A',
                        max_storage = 30,
                        max_release = 100.0,
                        hydraulic_head = 60,
                        efficiency = 0.8
                    )

    state = ReservoirState(reservoir=projectA, storage=10)

    #register the dynamic listener
    state.on_trait_change(wake_up_watchman_if_spillage, name='spillage')

    state.release = 90
    state.inflows = 0
    state.print_state()

    print 'Forcing spillage'
    state.inflows = 100
    state.release = 0

    print 'Why do we have two executions of the callback ?'
