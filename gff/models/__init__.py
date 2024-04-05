from . import debug_model


def create(C, n_hydroatlas):
    if C["model"] == "debug":
        return debug_model.DebugModel(n_hydroatlas, C["hydroatlas_dim"])
