from heapq import heapify, heappush, heappop
import torch
from copy import deepcopy

class priority_dict(dict):
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

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._heap = [(k, v) for k, v in self.items()]
        self._rebuild_heap()

    def sethyper(self,decay_rate = 0.5, K = 5):
        self.k = K
        self.decay_rate = decay_rate

    def _reorder(self):
        #super(priority_dict, self).__init__(self._heap[-self.k:])
        self._heap = deepcopy(self._heap[-self.k:])
        in_heap = [k for k,_ in self._heap]
        del_ = [k for k in self.keys() if k not in in_heap]
        for k in del_:
            del self[k]


    def _rebuild_heap(self):
        #self._heap = [(k, v) for k, v in self.items()]
        heapify(self._heap)
        if not self.isEmpty() and self.isFull():
            self._reorder()

    def isEmpty(self):
        if len(self._heap) == 0:
            return True
        return False

    def decay(self):
        self._heap = [(self.decay_rate*k, v) for k, v in self._heap]

    def isFull(self):
        if len(self._heap) < self.k:
            return False
        return True

    def pokesmallest(self):
        """Return the lowest priority.

        Raises IndexError if the object is empty.
        """

        k,_ = self._heap[0]
        return k

    def gradsum(self):
        """Return the sum of top k gradients
        """

        sum = torch.clone(self._heap[0][1])
        for _,v in self._heap[1:]:
            sum.add_(v)
        return sum

    def __getitem__(self,key):

        return dict(self._heap)

    def __len__(self):
        return len(self._heap)

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        #super(priority_dict, self).__setitem__(key, val)
        self._heap.append((key, val))
        self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()
