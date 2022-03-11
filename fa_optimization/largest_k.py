import heapq

class LargestK:
    'Keep track of the largest k items'
    def __init__(self, k):
        self.heap = []
        self.k = k
        self.scores = set()
    
    def can_add(self, score):
        if score in self.scores:
            return False
        if len(self.heap) < self.k or score > self.heap[0][0]:
            return True
        return False
        
    def add(self, score, thing, copy=False):
        'thing should have a copy method!'
        if not self.can_add(score):
            return
        if copy:
            thing = thing.copy()
        if len(self.heap) == self.k:
            self.scores.remove(self.heap[0][0])
            heapq.heapreplace(self.heap, (score,thing))
        else:
            heapq.heappush(self.heap, (score, thing))
        self.scores.add(score)
    
    def items(self, no_scores=False):
        if no_scores:
            return [ i[1] for i in self.heap ]
        return self.heap
    
    def largest(self):
        if len(self.heap)==0:
            return -1, None
        return heapq.nlargest(1, self.heap)[0]

