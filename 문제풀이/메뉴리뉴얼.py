import itertools
import collections


def solution(orders, course):
    answer = []
    for c in course:
        temp = []
        for order in orders:
            combi = itertools.combinations(sorted(order), c)
            temp += combi
        counter = collections.Counter(temp)

        if len(counter) != 0 and max(counter.values()) != 1:
            answer += [''.join(f) for f in counter if counter[f] == max(counter.values())]
    return sorted(answer)