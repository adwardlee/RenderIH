from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class DataProvider:
    def __init__(self, dataset, batch_size, num_workers=1, dist=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataiter = None
        self.iteration = 0
        self.epoch = 0
        self.num_workers = num_workers
        self.dist = dist
        self.build()

    def build(self):
        if self.dist:
            sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=True)
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                                    num_workers=self.num_workers, drop_last=True, pin_memory=True)
        else:
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers, drop_last=True, pin_memory=True)
        self.batch_per_epoch = len(dataloader)
        self.dataiter = dataloader.__iter__()

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            self.iteration += 1
            return next(self.dataiter)#self.dataiter.next()

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration = 1
            return next(self.dataiter)#self.dataiter.next()
