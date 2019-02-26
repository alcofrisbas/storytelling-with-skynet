import random
money = [4, 2, 2, 1, 10]
weight = [12,6, 1, 3, 14]

capacity = 15

def eval_fitness(l):
    total_money = 0
    total_weight = 0
    for i in range(len(l)):
        if l[i] == "1":
            total_money += money[i]
            total_weight += weight[i]
    if total_weight > capacity:
        return 0
    return total_money

def reproduce(a, b):
    k = random.randrange(len(a))
    c = a[:k]+b[k:]
    d = b[:k]+a[k:]
    return c, d

def create_population(n):
    p = []
    for i in range(n):
        p.append("".join([str(random.randrange(2)) for j in range(5)]))
    return(p)

if __name__ == '__main__':
    population = create_population(30)
    while len(population) > 3:
        fit = {}
        for p in population:
            fit[p] = eval_fitness(p)
        pop_list = sorted(fit.items(),  key=lambda x: x[1], reverse=True)
        for i in range(1):
            pop_list.pop()

        if len(pop_list) % 2 != 0:
            pop_list.pop()
        population = []
        for i in range(0, len(pop_list),2):
            c,d = reproduce(pop_list[i][0], pop_list[i+1][0])
            population.append(c)
            population.append(d)

    print(population)
