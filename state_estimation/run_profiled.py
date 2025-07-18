import cProfile
import runpy

cProfile.run("runpy.run_module('state_estimation.state_estimator', run_name='__main__')", filename="profile.prof")

# view profiler results with: snakeviz profile.prof 