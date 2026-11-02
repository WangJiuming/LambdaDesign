"""LambdaDesign inference for the HeLa/mRNA model."""

__version__ = "1.0.1"


def __getattr__(name):
    if name == "LambdaDesign":
        from .inference import LambdaDesign
        return LambdaDesign
    if name == "GenerationOptions":
        from .generation import GenerationOptions
        return GenerationOptions
    raise AttributeError(name)


__all__ = ["LambdaDesign", "GenerationOptions", "__version__"]
