import numpy
import sys

def markov_decision_process(o, g, h, s, T, f, p, test=False):
    def cal_transition_function():
        """
        Return P where P[before_sold][after_sold] is the probablity of given before_sold products at the begining of the month, there is after_sold products left at the end if the month. See table 1
        """
        if test:
            return numpy.array(
                [
                    [1, 0, 0, 0],
                    [3/4, 1/4, 0, 0],
                    [1/4, 1/2, 1/4, 0],
                    [0, 1/4, 1/2, 1/4]
                ]
            )
        P = numpy.zeros((s+1, s+1))
        # First dimension is products amount before sold
        for before_sold in range(s+1):
            # Senond dimension is products amount after sold
            # But we just loop the demand for convenient, and add the probablity to correspond element in P
            for demand in range(s+1):
                after_sold = 0
                if(demand >= before_sold):
                    after_sold = 0
                else:
                    after_sold = before_sold - demand
                P[before_sold][after_sold] += p[demand]
        return P

    P = cal_transition_function()
    print(f'Transition probablity P[before_sold][after_sold] =\n{P}\n')

    def cal_reward_matrix():
        """
        Return R where R[current_stock][action] is the reward got if the decision maker take action `action` when current stock is `current_stock`. See table 2
        """
        if test:
            return numpy.array(
                [
                    [0, -1, -2, -5],
                    [5, 0, -3, 0],
                    [6, -1, 0, 0],
                    [5, 0, 0, 0]
                ]
            )

        def F(stock):
            """
            Return corresponding F accoring f and p. See equation (14)
            """
            reward = 0
            for demand in range(s+1):
                if demand <= stock:
                    reward += p[demand] * f(demand)
                else:
                    # Sold out all stock
                    reward += p[demand] * f(stock)
            return reward

        R = numpy.zeros((s+1, s+1))
        # First dimension is products amount before taking action
        for before_action in range(s+1):
            # Second dimension is the action's inbound products amount
            for inbound in range(s+1):
                # no need to calculate ❌ in table 2
                if(before_action+inbound > s):
                    continue
                R[before_action][inbound] = F(before_action+inbound) - o(inbound) - h(before_action+inbound)
        return R

    R = cal_reward_matrix()
    print(f'Reward matrix R[current_stock][action] =\n{R}\n')

    def cal_cumulative_reward():
        """
        Return a tuple (u, a), See equation (12)
        where u[time][current_stock] is the cumulative maximum reward when there are `current_stock` stock at time `time`. 
        and where a[time][current_stock] is the best action to take when stock amount is `current_stock` at time `time`.
        """
        u = numpy.zeros((T+1, s+1))
        a = numpy.zeros((T+1, s+1))
        # First dimension is time, in range [0, 1, ..., T] 
        # We should initialize u[T] first
        for stock in range(s+1):
            u[T][stock] = g(T)
        # Then we begin to get u[T-1], u[T-2], ..., u[0]
        for t in range(T-1, -1, -1):
            print('Calculating time:', t)
            # Second dimension is current stock
            for current_stock in range(s+1):
                print('\t if current stock is:', current_stock)
                best_action = 0
                maximum_reward = 0
                # For all actions, we calculate a cumulative reward, and then take the maxmimum, see equation (16)
                for inbound in range(0, s+1-current_stock):
                    # reward from current_stock to next_stock
                    reward = R[current_stock][inbound]
                    # reward after current_stock
                    for next_stock in range(s+1):
                        reward += P[current_stock+inbound][next_stock] * u[t+1][next_stock]
                    print('\t\t if take action:', inbound, 'the cumulative reward will be:', reward)
                    if reward > maximum_reward:
                        maximum_reward = reward
                        best_action = inbound
                u[t][current_stock] = maximum_reward
                a[t][current_stock] = best_action
                print('\t so we choose action:', best_action, 'and will get max reward:', maximum_reward)
            print('\n')
        return (u, a)
        
    (u, a) = cal_cumulative_reward()
    print(f'Cumulative maximum reward u[time][current_stock] =\n{u}\n')
    print(f'Best action a[time][current_stock] = \n{a}\n')



def run_example(T=3, test=False):
    # Model parameters, see equation (13)
    def o(u):
        return 2*u;
    def g(u):
        return 0;
    def h(u):
        return u;
    def f(u):
        return 8*u;
    s = 3
    # p_0, p_1, ..., p_s-1, q_s
    # p_0, p_1, p_2, p_3 for this example
    p = [1/4, 1/2, 1/4, 0]

    markov_decision_process(o, g, h, s, T, f, p, test=test)


print('T =', sys.argv[1], '\n')
run_example(T=int(sys.argv[1]), test=(len(sys.argv)>=3 and (sys.argv[2] == 'test')))