# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#   OpenBench is a chess engine testing framework authored by Andrew Grant.   #
#   <https://github.com/AndyGrant/OpenBench>           <andrew@grantnet.us>   #
#                                                                             #
#   OpenBench is free software: you can redistribute it and/or modify         #
#   it under the terms of the GNU General Public License as published by      #
#   the Free Software Foundation, either version 3 of the License, or         #
#   (at your option) any later version.                                       #
#                                                                             #
#   OpenBench is distributed in the hope that it will be useful,              #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of            #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
#   GNU General Public License for more details.                              #
#                                                                             #
#   You should have received a copy of the GNU General Public License         #
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import math


def erf_inv(x):
    a = 8*(math.pi-3)/(3*math.pi*(4-math.pi))
    y = math.log(1-x*x)
    z = 2/(math.pi*a) + y/2
    return math.copysign(math.sqrt(math.sqrt(z*z - y/a) - z), x)

def phi_inv(p):
    return math.sqrt(2)*erf_inv(2*p-1)


def bayeselo_to_proba(elo, drawelo):
    pwin  = 1.0 / (1.0 + math.pow(10.0, (-elo + drawelo) / 400.0))
    ploss = 1.0 / (1.0 + math.pow(10.0, ( elo + drawelo) / 400.0))
    pdraw = 1.0 - pwin - ploss
    return pwin, pdraw, ploss

def proba_to_bayeselo(pwin, pdraw, ploss):
    elo     = 200 * math.log10(pwin/ploss * (1-ploss)/(1-pwin))
    drawelo = 200 * math.log10((1-ploss)/ploss * (1-pwin)/pwin)
    return elo, drawelo


def SPRT(wins, losses, draws, elo0, elo1):
    adj_wins = wins + 1
    adj_losses = losses + 1
    adj_draws = draws + 1

    N = adj_wins + adj_losses + adj_draws
    elo, drawelo = proba_to_bayeselo(float(adj_wins)/N, float(adj_draws)/N, float(adj_losses)/N)

    # Probability laws under H0 and H1
    p0win, p0draw, p0loss = bayeselo_to_proba(elo0, drawelo)
    p1win, p1draw, p1loss = bayeselo_to_proba(elo1, drawelo)

    # Log-Likelyhood Ratio
    return    adj_wins * math.log(p1win  /  p0win) \
          + adj_losses * math.log(p1loss / p0loss) \
          +  adj_draws * math.log(p1draw / p0draw)

def ELO(wins, losses, draws):

    def _elo(x):
        if x <= 0 or x >= 1: return 0.0
        return -400*math.log10(1/x-1)

    # win/loss/draw ratio
    N = wins + losses + draws;
    if N == 0: return (0, 0, 0)
    w = float(wins)  / N
    l = float(losses)/ N
    d = float(draws) / N

    # mu is the empirical mean of the variables (Xi), assumed i.i.d.
    mu = w + d/2

    # stdev is the empirical standard deviation of the random variable (X1+...+X_N)/N
    stdev = math.sqrt(w*(1-mu)**2 + l*(0-mu)**2 + d*(0.5-mu)**2) / math.sqrt(N)

    # 95% confidence interval for mu
    mu_min = mu + phi_inv(0.025) * stdev
    mu_max = mu + phi_inv(0.975) * stdev

    return (_elo(mu_min), _elo(mu), _elo(mu_max))

def sort_by_puct(tests):
    total_visits = sum(test.games for test in tests)
    sqrt_total_visits = math.sqrt(total_visits)
    explore_coef = 2.25 * sqrt_total_visits

    def puct(test):
        llr = test.currentllr if test.test_mode == "SPRT" else 0
        sum_rewards = llr * test.games
        child_visits = test.games
        policy = test.priority

        return (sum_rewards + explore_coef * policy) / (child_visits + 1)

    return sorted(tests, key=lambda t: -puct(t))

