#------------------------------------------------------------------------------
#
# Author: Vinhthuy Phan, 2021
#
#------------------------------------------------------------------------------
import numpy


class Buckets:
    def __init__(self, v, min_amount, max_amount, min_diff=100):
        self.value = v
        self.len = len(v)
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.min_diff = min_diff
        if v[0] < v[-1]:
            self.order = 'increasing'
            if v[1] < min_amount:
                raise Exception('Bucket amount {} is smaller than allowed ({})'.format(v[1],min_amount))
            if v[-1] > max_amount:
                raise Exception('Bucket amount {} is larger than allowed ({})'.format(v[-1],max_amount))
        else:
            self.order = 'decreasing'
            if v[-2] < min_amount:
                raise Exception('Bucket amount {} is smaller than allowed ({})'.format(v[-2],min_amount))
            if v[0] > max_amount:
                raise Exception('Bucket amount {} is larger than allowed ({})'.format(v[0],max_amount))


    #--------------------------------------------------------------------------
    def is_monotonic(self):
        if self.order == 'increasing' and sorted(self.value)==self.value:
            return True
        if self.order == 'decreasing' and sorted(self.value, reverse=True)==self.value:
            return True
        return False

    #--------------------------------------------------------------------------
    def undo(self, which, amount, idx):
        if idx<0 or amount==0:
            return
        if which == 'add':
            self.value[idx] -= amount
        elif which == 'remove':
            self.value[idx] += amount
        else:
            raise Exception('Unknown move: ' + which)

    #--------------------------------------------------------------------------
    # add an amount to a random bucket, ensuring that amounts are monotonic
    #--------------------------------------------------------------------------
    def random_add(self, amounts):
        amount = numpy.random.choice(amounts)
        random_idx = numpy.arange(self.len)
        numpy.random.shuffle(random_idx)
        for idx in random_idx:
            # print('trying to add {} to bucket {} with current value {}'.format(amount,idx,self.value[idx]))
            if self.order == 'increasing':
                if idx==0:
                    continue
                if idx==self.len-1:
                    threshold = self.max_amount
                else:
                    threshold = self.value[idx+1] - self.min_diff
                if self.value[idx]+amount <= threshold:
                    self.value[idx] += amount
                    return amount, idx
            else:
                if idx==self.len-1:
                    continue
                if idx==0:
                    threshold = self.max_amount
                else:
                    threshold = self.value[idx-1] - self.min_diff
                if self.value[idx]+amount <= threshold:
                    self.value[idx] += amount
                    return amount, idx
        return 0, -1

    #--------------------------------------------------------------------------
    # remove an amount to a random bucket, ensuring that amounts are monotonic
    #--------------------------------------------------------------------------
    def random_remove(self, amounts):
        amount = numpy.random.choice(amounts)
        random_idx = numpy.arange(self.len)
        numpy.random.shuffle(random_idx)
        for idx in random_idx:
            # print('trying to remove {} from bucket {} with current value {}'.format(amount,idx,self.value[idx]))
            if self.order == 'increasing':
                if idx==0:
                    continue
                if idx==1:
                    threshold = self.min_amount
                else:
                    threshold = self.value[idx-1] + self.min_diff
                if self.value[idx]-amount >= threshold:
                    self.value[idx] -= amount
                    return amount, idx
            else:
                if idx==self.len-1:
                    continue
                if idx==self.len-2:
                    threshold = self.min_amount
                else:
                    threshold = self.value[idx+1] + self.min_diff
                if self.value[idx]-amount >= threshold:
                    self.value[idx] -= amount
                    return amount, idx
        return 0, -1
    
    #--------------------------------------------------------------------------
    def copy(self):
        return Buckets(self.value.copy(), self.min_amount, self.max_amount, self.min_diff)
    
    #--------------------------------------------------------------------------
    def lookup(self, c):
        if (c<0) or (c>=len(self.value)):
            raise Exception('Bucket out of range: {}'.format(c))
        return self.value[c]
    
    #--------------------------------------------------------------------------
    def __repr__(self):
        output = '{}, {}-{}, ['.format(
            self.order, self.min_amount, self.max_amount, self.min_diff)
        return output+', '.join([ str(round(b,2)) for b in self.value ]) + ']'

