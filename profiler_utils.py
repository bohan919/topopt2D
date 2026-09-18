'''
Code profiler for comparing the performance of different implementations
'''

import time
import cProfile
import pstats
import functools

def profile_optimisation(name, print_stats=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"========== Starting {name} ==========")
            start_time = time.perf_counter()
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Execute the topology optimisation loop
            x_final = func(*args, **kwargs)
            
            profiler.disable()
            end_time = time.perf_counter()
            
            if print_stats:
                stats = pstats.Stats(profiler).sort_stats('cumtime')
                print(f"\n--- Top 10 Bottlenecks in {name} ---")
                stats.print_stats(10)
                
            print(f"========== {name} Completed in {end_time - start_time:.4f} seconds ==========\n")
            return x_final
        return wrapper
    return decorator