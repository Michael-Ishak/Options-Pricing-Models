import math
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt

# To prevent arbitrage, 3C - 100 + 40 = 0 must hold. Therefore, C must be $20.
# Also determined from binomial theory: if Su = price at upstate, Sd = price at downstate, r = 1 + interest, and p = risk neutral probability of an up move
# u = 2 * S, d = 0.5 * S. Here, S0 is 1

r = 1 + 0.25
d = 0.5
u = 2

print("Therefore, p value is: ", (r - d / u - d))

# Given that we are not given the true values of u and d, we can estimate it using the asset's volatility.

N = 4
S0 = 100
T = 0.5
sigma = 0.4
dt = T / N
K = 105
r = 0.05
u = math.exp(sigma * math.sqrt(dt))
d = math.exp(-sigma * math.sqrt(dt))
p = (math.exp(r * dt) - d) / (u - d)

# Finding terminal stock prices for a 4 step model (N = 4)
print("Finding terminal stock prices for a 4 step model (N = 4):")

for k in reversed(range(N + 1)):
    ST = S0 * u**k * d ** (N - k)
    print(round(ST, 2), round(max(ST - K, 0), 2))

# To find the probability at each node (p_star), we do:

print("Finding node probabilities:")


def combos(n, i):
    return math.factorial(n) / (math.factorial(n - i) * math.factorial(i))


for k in reversed(range(N + 1)):
    p_star = combos(N, k) * p**k * (1 - p) ** (N - k)
    print(round(p_star, 2))

# The value of the call is the weighted average of each p_star value multiplied by its corresponding probability value

# Valuing the call from the Cox 1979 paper example:
C = 0
for k in reversed(range(N + 1)):
    p_star = combos(N, k) * p**k * (1 - p) ** (N - k)
    ST = S0 * u**k * d ** (N - k)
    C += max(ST - K, 0) * p_star

print(math.exp(-r * T) * C)

# Putting this into a single function for simplicity:

print(
    "When putting everything together, this is the price we get from the Binomial Pricing Model:"
)


def binom_EU1(S0, K, T, r, sigma, N, type_="call"):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    p = (math.exp(r * dt) - d) / (u - d)
    value = 0
    for i in range(N + 1):
        node_prob = combos(N, i) * p**i * (1 - p) ** (N - i)
        ST = S0 * (u) ** i * (d) ** (N - i)
        if type_ == "call":
            value += max(ST - K, 0) * node_prob
        elif type_ == "put":
            value += max(K - ST, 0) * node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'")

    return value * math.exp(-r * T)


# Checking to see if we get the same value
binom_EU1(S0, K, T, r, sigma, N)

# If we simulate a heads and tails game for the stock option, we can simulate the asset's price between the current date and the expiration, with the limit of dt -> 0

N = 100000
sigma = 0.4
T = 0.5
K = 105
r = 0.05
dt = T / N
Heads = math.exp(sigma * math.sqrt(dt))
Tails = math.exp(-sigma * math.sqrt(dt))
S0 = 100
p = (math.exp(r * dt) - Tails) / (Heads - Tails)
paths = np.random.choice([Heads, Tails], p=[p, 1 - p], size=(N, 1))
plt.plot(paths.cumprod(axis=0) * 100, color="black")
plt.xlabel("Steps")
plt.ylabel("Stock Price")
plt.savefig("graph")

# YF real stock options



# def get_data(symbol, expiration_date, type == "call"):
#     option = symbol.option_chain(date = "{}".format(expiration_date))

#     if type == "call":
#         option.calls
#     if type == "put":
#         option.puts

#     obj = web.YahooOptions(f'{symbol}')
    
#     df = obj.get_all_data()

    # df.reset_index(inplace=True)

    # df['mid_price'] = (df.Ask+df.Bid) / 2
    # df['Time'] = (df.Expiry - dt.datetime.now()).dt.days / 255

    # return df[(df.Bid>0) & (df.Ask >0)]


# df = get_data('TSLA')

# prices = [] 


# for row in df.itertuples():
#     price = binom_EU1(row.Underlying_Price, row.Strike, row.Time, 0.01, 0.5, 20, row.Type)
#     prices.append(price)


# df['Price'] = prices
    
# df['error'] = df.mid_price - df.Price 


# exp1 = df[(df.Expiry == df.Expiry.unique()[2]) & (df.Type=='call')]


# plt.plot(exp1.Strike, exp1.mid_price,label= 'Mid Price')
# plt.plot(exp1.Strike, exp1.Price, label = 'Calculated Price')
# plt.xlabel('Strike')
# plt.ylabel('Call Value')
# plt.legend()