'''
5. (Bayes’ theorem) Assume that on average fraternal twins (two fertilized eggs) occur once in 150 births and identical twins (single egg divides into two separate embryos) once in 400 births (Note! This is not the true values, see Exercise 1.5 in BDA3). American male singer-actor Elvis Presley (1935 – 1977) had a twin brother who died in birth. What is the probability that Elvis was an identical twin? Assume that an equal number of boys and girls are born on average.
Implement this as a function in R that computes the probability. Below is an example of how the functions should be named and work if you want to check your result with markmyassignment.
'''


def calculate_iden_twin():
    p_b, p_a = 1 / 800, 1 / 600
    p_t_a = p_t_b = 1

    p_t = p_t_a * p_a + p_t_b * p_b

    p_b_t = round(p_b * p_t_b / p_t, 4)

    print('The probability that Elvis was an identical twin is {}'.format(p_b_t))

calculate_iden_twin()
