import util

priorityQueue = util.PriorityQueue()

priorityQueue.push(((34, 11), []), 4)

priorityQueue.push(((12, 11), []), 11)


for temp in priorityQueue.heap:
    print(temp)


queue = util.Queue()
queue.push(((12, 11), ['West', 'West', 'West', 'West', 'West', 'North', 'West']))
queue.push(((34, 22), ['West', 'South', 'South', 'West', 'West', 'West', 'North']))
queue.push(((66, 55), ['West', 'West', 'South', 'North', 'South', 'West', 'West']))
queue.push(((88, 66), ['West', 'North', 'West', 'South', 'South', 'West', 'South']))

for state in queue.list:
    print(state[0])


print("====================")
from util import heappush, heappop
openset = []
heappush(openset, (5, "foo"))
heappush(openset, (7, "bar"))
heappush(openset, (3, "baz"))
heappush(openset, (9, "quux"))
best = heappop(openset)
print(best)