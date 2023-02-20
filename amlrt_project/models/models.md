# Structure of the `models` module

This module contains the *task* and the *models* used for that task, as well as the losses and optimizers used to train models.

# Factories
Other than the task, the components are instantiated by factories. Factories are responsible for the details of object construction, and have data members for the configuration of those objects. They are responsible for moving configuration around.

# Classes to implement

You will need to implement at least two classes:
1. The task class, `ImageClassification` in this demo.
2. The model class, `SimpleMLP` in this demo.
2.1. You will also need to create a model factory.

You might also need to implement a loss function, with the corresponding factory, as well.

```mermaid
    ---
    Task components
    ---
    classDiagram
        LightningModule <|- Task <|.. ImageClassification
        Module <|- Model <|.. SimpleMLP
        Module <|- Loss <|.. CrossEntropy
        class Task
        class Loss
        class Model
        <<Abstract>> Task
        <<Abstract>> Model
        <<Abstract>> Loss
```

# Task creation

The main entry point is the `get_model` functions, which will parse the configuration into factories and use them to create the task with its components.