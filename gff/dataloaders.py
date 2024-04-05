import torch
import torch.nn
import torch.utils.data


class DebugFloodForecastDataset(torch.data.utils.Dataset):
    def __init__(self, n_examples, data_sources, *args, **kwargs):
        self.n_examples = n_examples
        self.n_times = 8
        if "era5_land" in data_sources:
            self.era5_land = torch.randn((self.n_examples, self.n_times, 5, 32, 32))
        if "hydroatlas_basin" in data_sources:
            self.hydroatlas_basin = torch.randn((self.n_examples, 64, 32, 32))
        if "hydroatlas_river" in data_sources:
            self.hydroatlas_river = torch.randn((self.n_examples, 32, 32))
        if "dem_coarse" in data_sources:
            self.dem_coarse = torch.randn((self.n_examples, 32, 32))
        if "dem_fine" in data_sources:
            self.dem_fine = torch.randn((self.n_examples, 256, 256))
        if "s1" in data_sources:
            self.s1 = torch.randn((self.n_examples, 2, 256, 256))

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return {k: getattr(self, k)[idx] for k in self.data_sources}


class FloodForecastDataset(torch.data.utils.Dataset):
    def __init__(self, folder):
        raise NotImplementedError()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def create(C, generator):
    if C["dataset"] == "debug_dataset":
        train_ds = DebugFloodForecastDataset(C["data_sources"])
        test_ds = DebugFloodForecastDataset(C["data_sources"])
    elif C["dataset"] == "forecast_dataset":
        train_ds = FloodForecastDataset(
            C["data_folder"], split="train", test_partition=C["test_partition"]
        )
        test_ds = FloodForecastDataset(
            C["data_folder"], split="test", test_partition=C["test_partition"]
        )
    else:
        raise NotImplementedError(f"Dataset {C['dataset']} not supported")

    kwargs = {k: C[k] for k in ["batch_size", "num_workers"]}
    train_dl = torch.utils.data.DataLoader(train_ds, **kwargs, generator=generator)
    test_dl = torch.utils.data.DataLoader(test_ds, **kwargs, generator=generator)

    return train_dl, test_dl
