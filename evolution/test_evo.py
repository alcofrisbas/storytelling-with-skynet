from evo import *


def test_main():
    labels = ["init_scale","learning_rate", "max_grad_norm", "num_layers",
            "num_steps", "hidden_size", "max_epoch", "max_max_epoch", "keep_prob",
            "lr_decay", "batch_size"]

    values1 = [0.1, 1.0, 5, 2, 20, 200, 4, 1, 1.0, 0.5, 1]
    values2 = [0.05, 1.0, 5, 2, 35, 650, 6, 1, 0.8, 0.8, 1]
    exempt = {"vocab_size":10011}
    i1 = Individual(labels, values2, exempt= exempt)
    i2 = Individual(labels, values1,exempt = exempt)

    i3, i4 = reproduce(i1,i2, k=300)
    #for i in range(len(i3.labels)):
    #    print("{}:\t{:.5}\t{:.5}".format( i3.labels[i], str(i4.values[i]), str(i3.values[i])))
    file_from_population("test_pop", [i1,i2,i3,i4])
    pop = population_from_file("test_pop")
    print(pop[2])

if __name__ == '__main__':
    test_main()
