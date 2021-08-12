
def get_data_odor_num_val():
    # get data
    odor = ''
    data = []
    with open('mushrooms_data.txt') as f:
        lines = f.readlines()
    for line in lines:
        odor = line[10]
        if odor == 'a': # almond
            data.append(1)
        elif odor == 'l': # anise
            data.append(2)
        elif odor == 'c': # creosote
            data.append(3)
        elif odor == 'y':  # fishy
            data.append(-1)
        elif odor == 'f':  # foul
            data.append(-2)
        elif odor == 'm':  # musty
            data.append(-3)
        elif odor == 'n':  # none
            data.append(0)
        elif odor == 'p':  # pungent
            data.append(4)
        elif odor == 's':  # spicy
            data.append(5)

    data_size = len(data)
    learn_size = round(data_size*0.8)

    learn_data = data[0:learn_size]
    test_data = data[learn_size:]

    print(test_data)
