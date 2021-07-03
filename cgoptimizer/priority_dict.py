from heapq import heapify, heappush, heappushpop, nlargest
import torch
from copy import deepcopy


class HeapItem:
    def __init__(self, p, t):
        self.p = p
        self.t = t

    def __lt__(self, other):
        return self.p < other.p


class PriorityDict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, decay_rate=0.5, K=5, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self.k = K
        self.decay_rate = decay_rate
        self._heap = nlargest(self.k, [HeapItem(k, v) for k, v in self.items()])
        heapify(self._heap)
        self._sum = sum([it.t for it in self._heap])
        self._mean = self._sum / len(self._heap) if len(self._heap) > 0 else 0

    def size(self):
        return len(self._heap)

    def set_hyper(self, decay_rate=0.5, K=5):
        self.k = K
        self.decay_rate = decay_rate

    def is_empty(self):
        return len(self._heap) == 0

    def decay(self):
        for item in self._heap:
            item.p *= self.decay_rate

    def is_full(self):
        return len(self._heap) >= self.k

    def average_topC(self):
        ave = 0.0
        if len(self._heap) > 0:
            ave = sum([it.t.norm() for it in self._heap]) / float(len(self._heap))
        return ave

    def poke_smallest(self):
        """Return the lowest priority.

        Raises IndexError if the object is empty.
        """

        it = self._heap[0]
        return it.p

    def gradmean(self):
        """Return the sum of top k gradients"""

        return self._mean

    def gradsum(self):
        """Return the sum of top k gradients"""
        return self._sum

    def __getitem__(self, key):
        return dict(self._heap)

    def __len__(self):
        return len(self._heap)

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        if key >= 0:
            if len(self._heap) >= self.k:
                removed = heappushpop(self._heap, HeapItem(key, val)).t
            else:
                heappush(self._heap, HeapItem(key, val))
                removed = 0
            self._sum = self._sum + val - removed
            self._mean = self._sum / len(self._heap)

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html

        super(PriorityDict, self).update(*args, **kwargs)

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()
