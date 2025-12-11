import os.path as osp
import random
from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class LULC(DatasetBase):
    """LULC Dataset.

    Expected structure:
        root/
            lulc/
                satellite/
                    class1/
                        img1.jpg
                        ...
                    class2/
                        ...
    """

    dataset_dir = ""
    domains = ["EuroSAT_RGB"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        # We only have one domain "satellite", but we can simulate source/target split
        # or just use the same domain for both if we rely on transforms.
        # For IP-CLIP, we typically define source and target domains in config.
        # Here we assume the user puts everything under "satellite" domain.
        
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x, test_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train_x")
        train_u, test_u = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train_u")
        
        # For free mode testing, we might want to split the test set
        # But for now let's just load everything available
        test_1, test_2, test_3, test_4 = self._read_data_test(self.domains)

        super().__init__(
            train_x=train_x,
            train_u=train_u,
            test_x=test_x,
            test_u=test_u,
            test_1=test_1,
            test_2=test_2,
            test_3=test_3,
            test_4=test_4
        )

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            if not osp.exists(domain_dir):
                print(f"Warning: Domain directory {domain_dir} does not exist. Skipping.")
                continue
                
            # Handle nested folder structure: datasets/EuroSAT_RGB/EuroSAT_RGB
            nested_dir = osp.join(domain_dir, dname)
            if osp.exists(nested_dir) and osp.isdir(nested_dir):
                print(f"Detected nested structure. Using {nested_dir} as domain directory.")
                domain_dir = nested_dir
                
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                if not osp.isdir(class_path):
                    continue
                    
                imnames = listdir_nohidden(class_path)
                for imname in imnames:
                    if not imname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        continue
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)
        
        # Simple random split for train/test if not pre-defined
        # Assuming 80% train, 20% test
        random.shuffle(items)
        num_total = len(items)
        num_train = int(num_total * 0.8)
        
        items_train = items[:num_train]
        items_test = items[num_train:]
        
        return items_train, items_test

    def _read_data_test(self, input_domains):
        # Just return the same test set for all "test_n" for now, 
        # or split if we had multiple domains.
        # Since we only have "satellite", we can just return the test split of it.
        
        # Re-read to get a consistent test set or just reuse _read_data logic?
        # To be consistent with OfficeHome logic where test_1..4 map to domains:
        
        items_map = {}
        for domain, dname in enumerate(input_domains):
            items = []
            domain_dir = osp.join(self.dataset_dir, dname)
            if not osp.exists(domain_dir):
                items_map[dname] = []
                continue

            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                if not osp.isdir(class_path):
                    continue

                imnames = listdir_nohidden(class_path)
                for imname in imnames:
                    if not imname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        continue
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)
            items_map[dname] = items

        # Since we only have "satellite", test_1 is satellite. 
        # Others can be empty or duplicates.
        # Let's make test_1 be the satellite test set.
        
        # Use the first available domain for testing
        if len(input_domains) > 0:
            first_domain = input_domains[0]
            test_items = items_map.get(first_domain, [])
            random.shuffle(test_items)
            num_total = len(test_items)
            num_train = int(num_total * 0.8)
            items_test = test_items[num_train:]
        else:
            items_test = []

        return items_test, items_test, [], []
