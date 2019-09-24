def read_data():
    '''
    read data from the algae.txt
    :return: parameters,  (θ, n)
    '''

    with open('./algae.txt', 'r') as f:
        data = f.readlines()
        amount = 0  # the amount of 1 in the data
        for n in data:
            if n[0] == '1':
                amount += 1

    return amount, round(amount / len(data), 2), len(data)


read_data()
