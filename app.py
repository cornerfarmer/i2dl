import sys
sys.path.append('../')
import TaskPlan

def create_app():
    return TaskPlan.run([TaskPlan.Project("exercise_1", "Task", name="Exercise 1")], 1)