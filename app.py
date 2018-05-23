import sys
sys.path.append('../')
import TaskPlan

def create_app():
    return TaskPlan.run([
        TaskPlan.Project("exercise_1", "SoftmaxTask", name="Exercise 1.1", config_dir="config_1", result_dir="results_1"),
        TaskPlan.Project("exercise_1", "TwoLayerTask", name="Exercise 1.2", config_dir="config_2", result_dir="results_2"),
        TaskPlan.Project("exercise_1", "FeaturesTask", name="Exercise 1.3", config_dir="config_3", result_dir="results_3")
    ], 1)