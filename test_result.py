with open("Results/infersent_cic_MSRP", "r") as f:
    line = f.readlines()

a = [(x.rstrip().split()[1])for x in line]

print("InferSent_cic_MSRP MAX: ", max(a), " Epoch: ", a.index(max(a)))


