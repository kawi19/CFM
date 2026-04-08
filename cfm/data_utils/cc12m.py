import webdataset as wds
import torch

def _has_image_filter(sample):
    """Filter to ensure sample contains an image."""
    return sample.get("image", None) is not None

class CustomDataCollatorImg:
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        imgs = [i["image"] for i in batch]
        idxs = [i['__key__'] for i in batch]
        imgs = torch.stack(imgs)
        return imgs, idxs


class CC12MImg:
    def __init__(self):
        pass

    def get_wds_dataset(self, input_shards, transform, batch_size, collator=None):
        """
        return a dataset that returns an image, and text
        """

        pipeline = [
            wds.SimpleShardList(input_shards),
        ]
        def has_image(sample):
         return sample.get("image", None) is not None

        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pilrgb", handler=wds.ignore_and_continue),
                wds.rename(image="jpg;png;jpeg"),
                # wds.select(has_image),
                wds.select(_has_image_filter),
                wds.map_dict(image=transform),])

        dataset = wds.DataPipeline(*pipeline)
        return dataset


    def get_dataloader(self, dataset, batch_size=None, shuffle=False, num_workers=1):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=CustomDataCollatorImg(),   
        )
        return loader

class CustomDataCollatorImg:
    def __call__(self, batch):
        # 🔹 Drop any Nones
        batch = [s for s in batch if s is not None and s.get("image") is not None]

        if len(batch) == 0:
            return None  # This lets DataLoader skip this batch.

        imgs = [s["image"] for s in batch]
        idxs = [s.get("__key__", "") for s in batch]  # Safely handle missing key
        imgs = torch.stack(imgs)
        return imgs, idxs

def get_dataloader(dataset, batch_size=None, shuffle=False, num_workers=1):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=CustomDataCollatorImg(),   
            prefetch_factor=3,
        )
        return loader