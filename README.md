# Mega Millions Lottery Calculations

**Disclaimer: I am not promoting nor endorsing gambling. The results here are purely for mathmatical understanding**

## Introduction:
The Mega Millions is a popular lottery game played in 45 states. The rules of the game are simple:
- Pick 5 balls without repeat from 1-70
- Pick a "Mega Ball" from 1-25.

Here is an example of a Mega Million drawing: <br>
[![An example of the Mega Million Lottery Drawing](https://img.youtube.com/vi/Zf42ebKP3mA/0.jpg)](https://www.youtube.com/watch?v=Zf42ebKP3mA)

What we aim to do today is analyze this game and find compute some interesting results.

<figure>
  <img src="/images/lets-go-gambling.webp" alt="Guy gambling" style="width:50%">
  <figcaption>Source: <a href="https://tenor.com/view/lets-go-gambling-gif-3937562841362806777"> Tenor</a> </figcaption>
</figure>

## TLDR Results:
### Lottery Calculations:
| Event            	| Probability 	| Percentage   	|
|------------------	|-------------	|--------------	|
| Jackpot          	| 3.304962e-9 	| 3.304962e-7% 	|
| 5 Balls, No Mega 	| 7.931909e-8 	| 7.931909e-6% 	|
| 4 Balls, Mega    	| 1.074113e-6 	| 1.074113e-4% 	|
| 4 Balls, No Mega 	| 2.577870e-5 	| 2.577870-3%  	|
| 3 Balls, Mega    	| 6.874432e-5 	| 6.874432e-3% 	|
| 3 Balls, No Mega 	| 1.649837e-3 	| 1.649837e-1% 	|
| 2 Balls, Mega    	| 1.443607e-3 	| 1.443607e-1% 	|
| 1 Ball, Mega     	| 1.118796e-2 	| 1.118796%    	|
| Mega Ball Only   	| 2.729861e-2 	| 2.729861%    	|

### Analysis
- Probability of winning anything: 95.8324306242948%
- Probability of winning nothing: 4.1675693757052%
- Probability of winning $100.00 or more: 0.0095679757052%
- Sample mean is (average jackpot amount): $281,192,307.69
- With a $281 Million dollar jackpot, the expected return on every ticket is:
    - $1.18 with 0% tax.
    - $0.71 with 40% tax.
    - $0.47 with 60% tax.
- For the expected return to exceed the ticket price ($2), the jackpot must be greater than:
    - $530 Million with 0% tax.
    - $933 Million with 40% tax.
    - $1.4 Billion with 60% tax.
- Tickets needed for XX% chance of winning the jackpot:
    - 1,393,410,944 tickets for 99%. 
    - 906,434,717 tickets for 95.0%. 
    - 419,458,491 tickets for 75.0%. 
    - 277,246,982 tickets for 60.0%. 
    - 31,879,495 tickets for 10.0%.
- With multiple winners, with 500,000,000 tickets, the expected winnings would be:
    - $254,394,154.40 with 0% tax.
    - $152,636,492.64 with 40% tax.
    - $101,757,661.76 with 60% tax.


That was a lot of numbers.

<figure>
  <img src="/images/stewie_losing_mind.webp" alt="Baby going crazy." style="width:25%">
  <figcaption>Source: <a href="https://giphy.com/gifs/family-guy-fox-family-guy-foxtv-3o6ZtaiPZNzrmRQ6YM"> Giphy</a> </figcaption>
</figure>


## Lottery Calculations:
<details><summary style="font-weight:bold">Click here for the calculations</summary>
<h3> 5 Balls and Mega Ball (Jackpot)</h3>
Let us first compute the total number of possible combinations:

$$ {70 \choose 5} * \frac{1}{25} =  302,575,350 $$

We will be using this number as the total number of combinations from now on.
Since we assume that each combination is equally likely, we may compute probabilities by computing

$$ \frac{\text{Favorable Outcomes}}{\text{Total Outcomes}}$$

Obviously, there is only 1 way to win it all, and so the jackpot probability is

$$ \frac{1}{302,575,350} \approx 3.304962\mathrm{e}{-9} = 3.304962\mathrm{e}{-7}\%$$

Which is 1 in 302,575,350.

You know it's bad when you have to write the probability in scientific notation.

<h3> 5 Balls, no Mega Ball </h3>
The math in this case is very similar, except we choose the wrong mega ball and since there is 24 ways to pick the wrong mega ball, the probability of winning is

$$ \frac{24}{302,575,350} \approx 7.931909 \mathrm{e}{-8} = 7.931909 \mathrm{e}{-6}\%$$

Or approximately 1 in 12,607,306.25


<h3> 4 Balls, Mega Ball </h3>
Here is where it gets *slightly* more complicated (at least for me).

We need to compute the number of winning outcomes of picking exactly 4 correct balls out of 5.

How can we do this systematically?

Think of it this way:
Let us fix some selection of 5 balls out of 70. Next let us choose 4 out of 5 of that fixed selection and then pick one of the 65 incorrectly. Furthermore, pick the correct Mega Ball.
Hence, we get the following:

$$ \frac{{5 \choose 4} * {65 \choose 1 } * 1}{302,575,350} = \frac{5*65}{302,575,350} \approx 1.074113\mathrm{e}{-6} = 1.074113\mathrm{e}{-4} \%$$

Or approximately 1 in 931001.08

<h3> 4 Balls, No Mega Ball </h3>
Again, we can do the same math, but this time with picking the wrong mega ball.

$$ \frac{{5 \choose 4} * {65 \choose 1 } * 24}{302,575,350} = \frac{5*65 * 24}{302,575,350} \approx 2.577870\mathrm{e}{-5} = 2.577870\mathrm{e}{-3} \%$$

Or approximately 1 in 38791.71

<h3> 3 Balls, Mega Ball </h3>
We may continue with the same process, except this time, we need to choose 2 wrong balls.

This is not that much different from 4 except we need to choose 2 this time instead of 1.
Hence,

$$ \frac{{5 \choose 3} * {65 \choose 2 } * 1}{302,575,350} = \frac{10*2080}{302,575,350} \approx 6.874432\mathrm{e}{-5} = 6.874432\mathrm{e}{-3} \%$$

Or approximately 1 in 14,546.89

<h3> 3 Balls, No Mega Ball </h3>
Same procedure as above, just multiply by 24 because we choose the wrong mega ball.

$$ \frac{{5 \choose 3} * {65 \choose 2 } * 24}{302,575,350} = \frac{10*2080 * 24}{302,575,350} \approx 1.649837\mathrm{e}{-3} = 1.649837\mathrm{e}{-1} \%$$

Or approximately 1 in 606.12

<h3> 2 Balls, Mega Ball </h3>
You get the drill by now.

$$ \frac{{5 \choose 2} * {65 \choose 3 } * 1}{302,575,350} = \frac{10*43,680 * 1}{302,575,350} \approx 1.443607\mathrm{e}{-3} = 1.443607\mathrm{e}{-1} \%$$

Or approximately, 1 in 692.71

<h3> 1 Ball, Mega Ball </h3>

$$ \frac{{5 \choose 1} * {65 \choose 4 } * 1}{302,575,350} = \frac{5*677,040 * 1}{302,575,350} \approx 1.118796\mathrm{e}{-2} = 1.118796\%$$

Or approximately 1 in 89.38

<h3> Mega Ball Only. </h3>

$$ \frac{{5 \choose 0} * {65 \choose 5} * 1}{302,575,350} = \frac{1*8,259,888 * 1}{302,575,350} \approx 2.729861\mathrm{e}{-2} = 2.729861\% $$

Or approximately 1 in 36.63

In conclusion, you have terrible odds of winning anything, but lets compute some more interesting things.

These numbers were cross references with the official Mega Millions site [1] and Durango Bill's calculations [2]
</details>

## Computation of Interesting topics:
From this point forwards, I will assume that you know a little bit of probability theory, and a bit of python as there is a lot of notation that gets thrown around.

### How much would the jackpot have to be in order for the Mega Millions to be a good game to play?
Let us define a couple of things:
Let us define our event space of interest as 

$$ E = \\{ 5BM, 5B, 4BM,4B,3BM, 3B, 2BM, 1BM, M \\} \subset \mathcal{P}({\Omega})$$

With 5BM = 5 Balls, Mega Ball; 5B = 5 Balls, No Mega Ball, etc.

Next, let us define a random variable X over omega as the following:

$$
X =
    \begin{cases}
        w & \text{if } 5BM \\
        1,000,000 & \text{if } 5B\\
        10,000 & \text{if } 4BM\\
        500 & \text{if } 4B\\
        200 & \text{if } 3BM\\
        10 & \text{if } 3B\\
        10 & \text{if } 2BM\\
        4 & \text{if } 1BM\\
        2 & \text{if } 1M\\
    \end{cases}
$$

where w is the jackpot winning amount


```python
w = None

pmf:dict = {
    # w: 3.304962e-9,
    1_000_000: 7.931909e-8,
    10_000: 1.074113e-6,
    500: 2.577870e-5,
    200: 6.874432e-5,
    10: 1.649837e-3 + 1.443607e-3, # Because these are mapped to the same value, we clobber the old value, thus, we must add.
    4: 1.118796e-2,
    2: 2.729861e-2
}

# Add the jackpot value in there.
if w in pmf:
    pmf[w] += 3.304962e-9
else:
    pmf[w] = 3.304962e-9

def cvt_perc(num:float) -> str:
    return str(num * 100)+"%"

def cvt_money(num:float) -> str:
    return "${:,.2f}".format(num)

```

### How likely are you to even win anything?

Then since the events are mutrally disjoint, we want to compute:

$$ \sum_i \mathbb{P}(X = E_i)  \ \forall E_i \in E$$

If we ignore the fancy notation, we are basically just saying how likely is it to win anything?


```python
def compute_proba_winning() -> float:
    total_prob:float = 0
    for proba in pmf.values():
        total_prob += proba
    return total_prob

total_prob:float = compute_proba_winning()

print("Your odds of winning anything is",1-total_prob, "or",cvt_perc(1-total_prob))
print("Your odds of winning nothing is",total_prob, "or",cvt_perc(total_prob))
```

    Your odds of winning anything is 0.958324306242948 or 95.8324306242948%
    Your odds of winning nothing is 0.041675693757052 or 4.1675693757052%


### How likely are you to win "decent" money?
Then we want to compute 

$$ \mathbb{P}{(X \geq m)} = 1 - F(m)$$

where m is the amount of money you want to see how likely you are to win. I choose m to be $100, but you can choose any value you consider "decent"


```python
def compute_proba_gt_val(val:float) -> float:
    # We will ASSUME that the jackpot value, is 20 million if not defined
    # Source: https://powerball-megamillions.com/how-powerball-megamillions-jackpots-are-calculated
    proba:float = 0
    for winning_amt in pmf:
        if winning_amt is None and 20_000_000 >= val:
            proba += pmf[winning_amt]
        elif winning_amt is not None and winning_amt >= val:
            proba += pmf[winning_amt]
    return proba


AMOUNT_OF_MONEY_TO_WIN:float = 100.00 # This is m, I just wrote the variable slightly more verbosely.

proba: float = compute_proba_gt_val(AMOUNT_OF_MONEY_TO_WIN)
print("Your odds of winning",cvt_money(AMOUNT_OF_MONEY_TO_WIN),"or more is", proba, "or", cvt_perc(proba))
```

    Your odds of winning $100.00 or more is 9.5679757052e-05 or 0.0095679757052%


### What is the expected return?
Let us first assume some conditions to make our lives easier:
 - There is only one winner
 - You get all of the money straight up as a lump sum
 - All the winnings are taxed equally.

Well, suppose that the jackpot is the sample average of the last 26 games played, then let us compute the expected return. [4]

By definition, $\mathbb{E}[X]$ is the following:

$$ \mathbb{E}[X] = (1-t) \sum x \mathbb{P}(X=x) = (1-t) * [w * \frac{1}{302,575,350} + 1,000,000 * \frac{24}{302,575,350} + ... + 2 * \frac{8,259,888}{302,575,350}]$$

<br> where $t \geq 0$ is the tax rate and w is the jackpot winning amount which we can suppose for this case is the sample average of the last 26 games played.

Note that we pulled the probabilities from the derivation above.

Then, let us compute the expected return from playing.


```python
data_millions:float = (233,215,197,181,165,145,129,110,94,77,59,44,28,20,113,95,77,62,42,20,1269,1000,862,760,695,619)
sample_avg:float = (sum(data_millions) / len(data_millions)) * 1e6
print("Sample mean is (average jackpot amount):",cvt_money(sample_avg))

# I will not set w in the pmf because that is a shared variable and might ruin future calculations if
# you're not careful, so I will just hard compute is assuming that w is still None.
def compute_expectation(jackpot_amt:float, tax_val:float = 0) -> float:
    total:float = 0
    for value in pmf:
        if value is None:
            total += (jackpot_amt * pmf[value])
        else:
            total += (value * pmf[value])
    
    return total * (1 - tax_val)

print("With 0% tax and a",cvt_money(sample_avg),"jackpot, the expected return on every ticket is",compute_expectation(sample_avg,0))
print("With 40% tax and a",cvt_money(sample_avg),"jackpot, the expected return on every ticket is",compute_expectation(sample_avg,0.4))
print("With 60% tax and a",cvt_money(sample_avg),"jackpot, the expected return on every ticket is",compute_expectation(sample_avg,0.6))
```

    Sample mean is (average jackpot amount): $281,192,307.69
    With 0% tax and a $281,192,307.69 jackpot, the expected return on every ticket is 1.1763118256153846
    With 40% tax and a $281,192,307.69 jackpot, the expected return on every ticket is 0.7057870953692308
    With 60% tax and a $281,192,307.69 jackpot, the expected return on every ticket is 0.4705247302461539

That's a lot of money!

<figure>
  <img src="/images/azusa-money.webp" alt="Doll turning shocked" style="width:25%">
  <figcaption>Source: <a href="https://giphy.com/gifs/shocked-face-meme-PtZzHZzuSmPCWxS5MJ"> Giphy</a> </figcaption>
</figure>

### How big does the jackpot have to be in order for the lottery to be "worth" playing?

This begs the question, since the expected return with the sample average is so bad, hypothetically, how big *would* the jackpot have to be before the expected return is more than the ticket price? Let us first assume the same assumptions that we made above.

Then we want to solve for w such that $\mathbb{E}[X] > v$
<br>Where v is the "value" we want to be bigger than. In this case, since the ticket is worth 2 dollars, we set v = 2.

<br> Before we solve this inequality, let us first denote s (as in sum) as

$$ s = 1,000,000 * \frac{24}{302,575,350} + ... + v * \frac{8,259,888}{302,575,350}$$

Then with a little rearrangement of the definition, we get the following:

$$ (1-t) [\frac{w}{302,575,350} + s] > v \implies w > \frac{v(1-t)}{302,575,350}$$

Applying what we have discovered, we will show some results:



```python
def compute_gt_val(val:float, tax_val:float = 0) -> float:
    assert(val >= 0)
    sum_wo_jackpot:float = 0
    for value in pmf:
        if value is None:
            sum_wo_jackpot += pmf[value]
        else:
            sum_wo_jackpot += (value * pmf[value])
    
    jackpot:float = ((val / (1-tax_val)) - sum_wo_jackpot) * 302_575_350
    return jackpot

def print_gt_val(val, tax_val):
    tax_adj_jackpot:float = compute_gt_val(val,tax_val)
    print("Assuming",cvt_perc(tax_val), "tax the jackpot would have to be",
            cvt_money(tax_adj_jackpot),"in order for the expected return of X to be greater than",val)

print_gt_val(2,0)
print_gt_val(2,.40)
print_gt_val(2,.60)
```

    Assuming 0% tax the jackpot would have to be $530,420,053.88 in order for the expected return of X to be greater than 2
    Assuming 40.0% tax the jackpot would have to be $933,853,853.88 in order for the expected return of X to be greater than 2
    Assuming 60.0% tax the jackpot would have to be $1,438,146,103.88 in order for the expected return of X to be greater than 2


Hence, the answer to our (multi) million dollar question is **530 million** in order for the expected return of the lottery to be greater than 2 (the amount it takes to play).

If we (unfortunately) don't live in a tax-free haven, then that number jumps up to nearly **1.4 billion** with a 60% tax!!

<figure>
  <img src="/images/shocked.webp" alt="Doll turning shocked" style="width:25%">
  <figcaption>Source: <a href="https://giphy.com/gifs/shocked-face-meme-PtZzHZzuSmPCWxS5MJ"> Giphy</a> </figcaption>
</figure>

### How many tickets would you have to buy in order to have a 99% chance of winning?
Notice that this is a geometric distribution. Where we are trying to find out the probability that we get one success in k trials and we want to find the smallest k such that the cumulative probability is 0.99
<br>Note that the CDF of a geometric distribution is the following:

$$ F(x) = 1 - (1-p)^x $$

where F is the CDF of the geometric distribution.

Thus, the inverse (quantile function) would be:

$$ F^{-1}(o) = \left \lceil{\frac{\ln{(1-o)}}{\ln{(1-p)}}}\right \rceil  = \left \lceil{\log_{(1-p)}(1-o) }\right \rceil$$

where x is the number of trials, o is the probability of success we want. I choose 0.99, but you can change the value to whatever you want inside of the following cell.


```python
import math
def compute_geom_inverse(o:float) -> int:
    assert(0 < o < 1)
    return math.ceil(math.log(1-o) / math.log(1-pmf[w]))

def print_num_tickets_needed(confidence:float) -> None:
    result:int = compute_geom_inverse(confidence)
    print(f'You need to buy {result:,} lottery tickets to get a {confidence * 100}% chance of winning the jackpot. ')

print_num_tickets_needed(0.99)
print_num_tickets_needed(0.95)
print_num_tickets_needed(0.75)
print_num_tickets_needed(0.6)
print_num_tickets_needed(0.5)
print_num_tickets_needed(0.1)

```

    You need to buy 1,393,410,944 lottery tickets to get a 99.0% chance of winning the jackpot. 
    You need to buy 906,434,717 lottery tickets to get a 95.0% chance of winning the jackpot. 
    You need to buy 419,458,491 lottery tickets to get a 75.0% chance of winning the jackpot. 
    You need to buy 277,246,982 lottery tickets to get a 60.0% chance of winning the jackpot. 
    You need to buy 209,729,246 lottery tickets to get a 50.0% chance of winning the jackpot. 
    You need to buy 31,879,495 lottery tickets to get a 10.0% chance of winning the jackpot. 


### What if there are multiple winners?
If we factor into account that the jackpot depends on how many people are playing, and the odds of you winning decreases as you more people play,
How much money would you win if there are n tickets in play?

The real formula actually uses past information and interpolation to figure out the amount, and it's kind of subjective based on holiday etc.
Thus, we will consider a much simpler (but reasonable) model.

Let us consider:

$$ w = \text{S} + \text{(C * R * n)} $$

where
<br> S = Starting Amount (How much does the jackpot start at?)
<br> C = Ticket Price (How much does each ticket cost?)
<br> R = Ticket Ratio (How much of each ticket goes towards the jackpot)
<br> n = Number of tickets sold

Let us now define some new random variables:

$$ Z_1, Z_2, ..., Z_n \stackrel{iid}{\sim} Ber(p) $$

The motivation of these random variables is that we want to track if ticket i won the jackpot.

Let us now further define another random variable:

$$Y := Z_1 \frac{w}{\sum_{i=1}^nZ_i}$$

The motivation behind this random variable is tracking the amount that you would win divided by the number of other people who would win.

Finding $\mathbb{E}[Y]$ is hard because you will likely divide by zero as the denominator is 0 with probability $(1-p)^n$ and $p \approx 3.3\mathrm{e}{-9}$ is extremely small. Thus, we will condition on the fact that you did win.

Thus, we will assume (without loss of generality) that you own ticket $Z_1$. Let us now compute the expectation of 

$$ \mathbb{E}[Y | Z_1 = 1] $$

or in other words, assuming you win, what is jackpot amount that you should expect to see?

$$ \mathbb{E}[Y | Z_1 = 1] = \mathbb{E}\frac{w}{1 + \sum_{i=2}^n Z_i} = \ w \mathbb{E} \frac{1}{1 + \sum_{i=2}^n Z_i} $$

Fortunately for us, $\sum_{i=2}^n \sim Binom(n-1,p)$ and furthermore, this has a closed form solution according to Chao and Strawdermann [3].

$$ w \mathbb{E} \frac{1}{1 + \sum_{i=2}^n Z_i} = w * \frac{1-(1-p)^n}{np}$$


```python
try:
    import numpy as np
    numpy_installed = True
except ImportError:
    numpy_installed:bool = False

TICKET_PRICE:float = 2.00
TICKET_RATIO:float = 0.5
STARTING_AMOUNT:int = 20_000_000

num_tickets:int = 500_000_000

proba_jackpot:float = pmf[None]

new_w:float = STARTING_AMOUNT + TICKET_PRICE * TICKET_RATIO * num_tickets

print(f"With {num_tickets:,} tickets in play, under our model, the jackpot amount would be {cvt_money(new_w)}")

denominator:float = (num_tickets * proba_jackpot)

# %timeit (1-proba_jackpot) ** num_tickets
# %timeit np.power((1-proba_jackpot),num_tickets)

# Use numpy here because it's 3 orders of magnitude faster.
if numpy_installed:
    numerator:float = new_w * (1 - np.power((1 - proba_jackpot),num_tickets))
else:
    numerator:float = new_w * (1 - (1-proba_jackpot) ** num_tickets)

expected_winning_untaxed:float = numerator / denominator

print(f"Expected (raw, untaxed) jackpot winnings with {num_tickets:,} tickets in play is {cvt_money(expected_winning_untaxed)}")
print("With 40% tax, the expected winnings is",cvt_money(0.6 * numerator / denominator))
print("With 60% tax, the expected winnings is",cvt_money(0.4 * numerator / denominator))
```

    With 500,000,000 tickets in play, under our model, the jackpot amount would be $520,000,000.00
    Expected (raw, untaxed) jackpot winnings with 500,000,000 tickets in play is $254,394,154.40
    With 40% tax, the expected winnings is $152,636,492.64
    With 60% tax, the expected winnings is $101,757,661.76


## Conclusion:
Almost all our calculations show that playing Mega Millions is not a good idea due to the extremely low probability of winning, even with assumptions of no tax and a single winner.

## References:
[1] https://www.megamillions.com/
<br>[2] https://durangobill.com/MegaMillionsOdds.html
<br>[3] https://www.jstor.org/stable/2284399?seq=1
<br>[4] https://www.usamega.com/mega-millions/results Retrieved on March 11, 2025
