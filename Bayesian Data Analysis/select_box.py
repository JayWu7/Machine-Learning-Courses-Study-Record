'''
question:
We have three boxes, A, B, and C. There are
• 2 red balls and 5 white balls in the box A,
• 4 red balls and 1 white ball in the box B, and • 1 red ball and 3 white balls in the box C.
Consider a random experiment in which one of the boxes is randomly selected and from that box, one ball is randomly picked up. After observing the color of the ball it is replaced in the box it came from. Suppose also that on average box A is selected 40% of the time and box B 10% of the time (i.e. P (A) = 0.4).
a) What is the probability of picking a red ball?
b) If a red ball was picked, from which box it most probably came from?
'''


def pick_red_ball():
    p_a, p_b, p_c = 0.4, 0.1, 0.5
    p_r_a = 2 / (2 + 5)
    p_r_b = 4 / (4 + 1)
    p_r_c = 1 / (1 + 3)

    p_r = p_a * p_r_a + p_b * p_r_b + p_c * p_r_c
    print('The probability of picking a red ball is {}'.format(p_r))

# pick_red_ball()


def boxes_probability():
    p_a, p_b, p_c = 0.4, 0.1, 0.5
    p_r = 0.319285

    p_r_a = 2 / (2 + 5)
    p_a_r = round(p_a * p_r_a / p_r, 3)

    p_r_b = 4 / (4 + 1)
    p_b_r = round(p_b * p_r_b / p_r, 3)

    p_r_c = 1 / (1 + 3)
    p_c_r = round(p_c * p_r_c / p_r, 3)

    print('If a red ball was picked, the probabilities of this ball is from box A,B,C are {}, {}, {} respectively'.format(p_a_r,p_b_r,p_c_r))

boxes_probability()
