try:
    from .pipelines import DiffusionForcingPipeline
except ModuleNotFoundError:
    DiffusionForcingPipeline = None
