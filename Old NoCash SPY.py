
# coding: utf-8

# ## Functions

# In[2]:


# Old Strategy

import scipy.stats as st
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import pandas_datareader as dr
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive('True')
'exec(%matplotlib inline)'
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from forex_python.converter import get_rate
from datetime import date

# Basic data collection functions
def getPrice(ticker, start='2013-08-07', end='2019-10-17'):
    df = dr.data.get_data_yahoo(ticker, start, end)
    # df = dr.data.get_data_yahoo(ticker, start='2015-02-05', end='2019-10-17')
    df.info()
    return df['Adj Close']

def logRet(df):
    # Plots daily returns
    pri = df
    daily_returns = pri * 0
    for i in range(1, len(pri)):
        daily_returns[i] = np.log(pri[i]/pri[i-1])
    return daily_returns

# For a portfolio's returns, gives the index
def indx(ret):
    ind = ret * 0
    ind += 100
    for i in range(1, len(ret)):
        ind[i] = ind[i-1] * (1+ret[i])
    return ind

# Cumulative returns

def cumRet(rets):
    cumR = rets * 0
    for t in range(1, len(cumR)):
        cumR[t] = cumR[t-1] + rets[t]
    return cumR

def plotCumRet(cumR):
    fig = plt.figure()
    ax = fig.add_axes([1,1,1,1])
    ax.plot(cumR)
    ax.set_xlabel('Year')
    ax.set_ylabel('% Return')
    ax.set_title('Cumulative Return')
    plt.show()

def plotCumRets(rets):
    c = cumRet(rets)
    plotCumRet(c)

# Risk metric functions

def maxDrawdown(pri, window=252):
    Roll_Max = pri.rolling(window, min_periods=200).max()
    Daily_Drawdown = pri/Roll_Max - 1.0
    Daily_Drawdown *= -1
    Max_Daily_Drawdown = np.max(Daily_Drawdown)
    print("Max daily drawdown: {}%".format(round(Max_Daily_Drawdown, 2)))
    return Max_Daily_Drawdown

def plotDrawdown(pri, window=252):
    Roll_Max = pri.rolling(window, min_periods=200).max()
    Daily_Drawdown = pri/Roll_Max - 1.0
    Daily_Drawdown *= -1
    fig = plt.figure()
    ax = fig.add_axes([1,1,1,1])
    ax.plot(Daily_Drawdown)
    ax.set_xlabel('Year')
    ax.set_ylabel('%')
    ax.set_title('Drawdown')
    plt.show()

def valueAR(dailyRet, per): #Historical
    pci = np.percentile(dailyRet, per)
    print("VaR at {0}%: {1}%".format(per, round(-100*pci, 3)))
    return pci

def cVAR(dailyRet, per):
    pci = np.percentile(dailyRet, per)
    sumBad = 0
    t = 0
    for i in range(1, len(dailyRet)):
        if dailyRet[i] <= pci:
            t += 1
            sumBad += dailyRet[i]
    if t > 0:
        condVaR = sumBad / t
    else:
        condVaR = 0
    print("CVaR at {0}%: {1}%".format(per, round(-100*condVaR, 3)))
    return condVaR

def semiv(dayRet, port=False):
    m = np.mean(dayRet)
    low = []
    for i in range(1, len(dayRet)):
        if dayRet[i] <= m:
            low.append(dayRet[i])
    stand = np.std(low)
    if port:
        stand *= np.sqrt(21)
    print("SemiSD: {}".format(round(stand, 6)))
    return stand

# Use pricing data
def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for i in range(1, len(X)):
        if X[i] > peak:
            peak = X[i]
        dd = (peak - X[i]) / peak
        if dd > mdd:
            mdd = dd
    return mdd

# SD, SemiV, Drawdown, VaR, CVaR, Skewness, Kurtosis
def riskMetrics(ret, name = "portfolio", varP = 5, cvarP = 5):
    print("Risk metrics for {}".format(name))
    pri = indx(ret)
    print("Average return: {}%".format(round(100*np.mean(ret), 6)))
    sd = np.std(ret)
    print("SD: {}".format(round(sd, 6)))
    semiv(ret)
    dd = max_drawdown(pri)
    print("Max DD: {}%".format(round(100*dd, 2)))
    valueAR(ret, varP)
    cVAR(ret, cvarP)
    sk = st.skew(ret)
    print("Skewness: {}".format(round(sk, 6)))
    kurt = st.kurtosis(ret)
    print("Kurtosis: {}".format(round(kurt+3, 6)))
    print("Excess Kurtosis: {}".format(round(kurt, 6)))
    print("")

def cVARNP(dailyRet, per):
    pci = np.percentile(dailyRet, per)
    sumBad = 0
    t = 0
    for i in range(1, len(dailyRet)):
        if dailyRet[i] <= pci:
            t += 1
            sumBad += dailyRet[i]
    if t > 0:
        condVaR = sumBad / t
    else:
        condVaR = 0
    print("CVaR at {0}%: {1}%".format(per, np.round(float(-100*condVaR), 3)))
    return condVaR

def riskMetricsNP(ret, name = "portfolio", varP = 5, cvarP = 5, port=False):
    print("Risk metrics for {}".format(name))
    pri = indx(ret)
    print("Average return: {}%".format(round(100*np.mean(ret), 6)))
    sd = np.std(ret)
    # Note change for monthly to work for portfolio
    if port:
        print("SD: {}".format(np.round(sd*np.sqrt(21), 6)))
    else:
        print("SD: {}".format(np.round(sd, 6)))
    semiv(ret, port=True)
    dd = max_drawdown(pri)
    print("Max DD: {}%".format(np.round(float(100*dd), 2)))
    valueAR(ret, varP)
    cVARNP(ret, cvarP)
    sk = st.skew(ret)
    print("Skewness: {}".format(np.round(float(sk), 6)))
    kurt = st.kurtosis(ret)
    print("Kurtosis: {}".format(round(kurt+3, 6)))
    print("Excess Kurtosis: {}".format(np.round(float(kurt), 6)))
    print("")

def getCSVPrice(t, start='2018-1-1', end='2020-1-1'):
    prices = dfP.loc[start : end, t]
    prices = np.array(prices)
    return prices

def getCSVRet(t, start='2018-1-1', end='2020-1-1'):
    rets = dfRet.loc[start:end, t]
    rets = np.array(rets)
    return rets

def strip_replace(x):
    return x.strip("'").strip('[').strip(']').strip().replace("'", "").split(', ')


# # Tickers

# In[3]:


#Alternative Energy
sym_AltEn=["NLR","FAN","PBD","TAN"]
#Consumer
sym_Cons=["PBS","PBJ","PEJ","PMR","BJK","XHB","CARZ"]
#Energy
sym_Ener=["PXE","FRAK","PXJ"]
#Financials
sym_Fina=["KBWP","IAI","PSP","KBWR","KBWB"]
#Healtcare
sym_Heal=["IHI","IHF","PBE","PJP"]
#Industrials &amp;Infrastructure
sym_Indu=["GII","PIO","PPA","IYT","SEA"]
#Materials, Metals and Mining
sym_Mate=["HAP","GDX","GDXJ","KOL","SLX","XME"]
#Technology
sym_Tech=["PSJ","SKYY","PXQ","PNQI","PSI","SOCL","FONE","ROBO"]
#Utilities
sym_Util= ["GRID"]

#Combining all symbols
sym_all=sym_AltEn+sym_Cons+sym_Ener+sym_Fina+sym_Heal+sym_Indu+sym_Mate+sym_Tech+sym_Util


tickers = sym_all
tickers.append('VTV')
tickers.append('SPY')
weights = np.zeros(len(tickers))
prices = 10 * np.ones(len(tickers))

print(tickers, '\n\n', weights, '\n\n', prices)


# In[4]:


dfList = pd.read_csv(open('WML_Old.csv')).dropna()

try:
    dfP = pd.read_csv(open('PricesSPY.csv'))

except:
    dfP = pd.DataFrame()
    for i in range(len(tickers)):
        dfP[tickers[i]] = getPrice(tickers[i], '01-01-2011', date.today())
    dfP.to_csv('PricesSPY.csv')

try:
    dfRet = pd.read_csv(open('ReturnsSPY.csv')).dropna()

except:
    dfRet = pd.DataFrame()
    for i in range(len(tickers)):
        dfRet[tickers[i]] = logRet(dfP[tickers[i]])
    dfRet = pd.DataFrame.fillna(dfRet, 0)
    dfRet.to_csv('ReturnsSPY.csv')

dfList['StartF'] = pd.to_datetime(dfList['StartF'], format='%d/%m/%Y')
dfList['EndF'] = pd.to_datetime(dfList['EndF'], format='%d/%m/%Y')
dfList['StartH'] = pd.to_datetime(dfList['StartH'], format='%d/%m/%Y')
dfList['EndH'] = pd.to_datetime(dfList['EndH'], format='%d/%m/%Y')
dfList['StartS'] = pd.to_datetime(dfList['StartS'], format='%d/%m/%Y')
dfList['EndS'] = pd.to_datetime(dfList['EndS'], format='%d/%m/%Y')

dfRet['Date'] = pd.to_datetime(dfRet['Date'], format='%Y/%m/%d')
dfP['Date'] = pd.to_datetime(dfP['Date'], format='%Y/%m/%d')
dfRet.set_index('Date', inplace=True)
dfP.set_index('Date', inplace=True)


# # Full Run

# In[5]:


# Collect data for each ticker in each list of winners
winners = dfList['Winners']
losers = dfList['Losers']
startF = dfList['StartF']
endF = dfList['EndF']
startH = dfList['StartH']
endH = dfList['EndH']
startS = dfList['StartS']
endS = dfList['EndS']

cumWin = []
cumLoss = []

# Volatility scaling
# Aiming for 12% annual volatility, which is 3.46% monthly
targetSD = np.sqrt(12) / 100 # 0.0346

monthlyReturn = np.zeros(len(winners))
standardDev = np.zeros(len(winners))
numWin = 5
dailyReturns = []
momWeight = np.zeros(len(winners))
momSD = np.zeros(len(winners))
g_win = np.zeros(len(winners))
g_los = np.zeros(len(winners))
hedgeWeight = np.zeros(len(winners))
g_spy = np.zeros(len(winners))
g_val = np.zeros(len(winners))

for i in range(len(winners)):
# for i in range(2):
    listWin = strip_replace(winners[i])
    listLoss = strip_replace(losers[i])
    numWin = len(listWin)

    retWin = 0
    retLoss = 0
    # Tickers
    for j in range(numWin):
        r = getCSVRet(listWin[j], startH[i], endH[i])
        retWin += r/len(listWin)
        g_win[i] = sum(retWin)

        r_l = getCSVRet(listLoss[j], startH[i], endH[i])
        retLoss += r_l/len(listLoss)
        g_los[i] = sum(retLoss)

    # Momentum strategy return
    moRet = retWin - retLoss

    # Momentum scaling
    retWinScale = pd.Series()
    retLossScale = pd.Series()
    for j in range(numWin):
        pWin = getCSVPrice(listWin[j], startS[i], endS[i])
        if j == 0:
            retWinScale = 0 * pd.Series(pWin)
            retLossScale = 0 * pd.Series(pWin)
        retWinScale += getCSVRet(listWin[j], startS[i], endS[i])
        pLoss = getCSVPrice(listLoss[j], startS[i], endS[i])
        retLossScale += getCSVRet(listLoss[j], startS[i], endS[i])
    scaleRet = (retWinScale - retLossScale) / len(listWin)
    totalVar = 0
    for j in range(len(scaleRet)):
        retj = float(scaleRet[j])
        totalVar += retj * retj
    monthlyVar = totalVar / 6
    monthlySD = np.sqrt(monthlyVar)
    momSD[i] = monthlySD
    wMo = targetSD / monthlySD

    pSPY = getCSVPrice('SPY', startH[i], endH[i])
    retSPY = getCSVRet('SPY', startH[i], endH[i])
    g_spy[i] = sum(retSPY)
    pVal = getCSVPrice('VTV', startH[i], endH[i])
    retVal = getCSVRet('VTV', startH[i], endH[i])
    g_val[i] = sum(retVal)
    hedgeRet = retVal - retSPY

    momWeight[i] = wMo
    wMo *= .5
    # wHedge = .5
    wHedge = 1 - wMo
    hedgeWeight[i] = wHedge
    cash = 1-wHedge-wMo

    fullPort = wMo * moRet + wHedge * hedgeRet
    dailyReturns.append(fullPort)
    # exRet = get_rate('USD', 'GBP', endH[i]) / get_rate('USD', 'GBP', startH[i])
    # fullPort *= exRet # Gives GBP return
    monthRet = np.sum(fullPort)
    monthlyReturn[i] = monthRet
    standardDev[i] = np.std(fullPort)


# In[6]:


monthlyReturn.mean()*12/(standardDev.mean()*np.sqrt(252))


# ## Metrics

# In[7]:


def perform(rets=monthlyReturn, sdev=standardDev):
    if type(rets) == pd.DataFrame:
        dfrets.plot(kind='line', x='Date', y='Returns', title='Monthly Returns',
                    figsize=(10,6))
        cumu= pd.DataFrame(rets['Returns'].cumsum())
        cumu['Date'] = rets['Date']
        cumu.plot(kind='line', x='Date', y='Returns', title='Cumulative Return',
                  figsize=(10,6))
    if type(sdev) == pd.DataFrame:
        dfstd.plot(kind='line', x='Date', y='Standard Deviation',
                   title='Daily Standard Deviation', figsize=(10,6))

    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(100 * rets)
        ax.set_xlabel('Month')
        ax.set_ylabel('Percent')
        ax.set_title('Monthly Returns')
        plt.show()

        riskMetricsNP(rets, 'Whole Portfolio', port=True)
        print('Cash: {}'.format(np.round(cash, 12)))

        fig2, ax2 = plt.subplots()
        ax2.plot(sdev)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('SD')
        ax2.set_title('Portfolio Standard Deviation')
        plt.show()

        plotCumRets(np.array(rets['Returns']))


dfrets = pd.DataFrame(monthlyReturn)
dfrets['Date'] = startH
dfrets['Date'] = pd.to_datetime(dfrets['Date'])
dfrets.columns = ['Returns', 'Date']
dfstd = pd.DataFrame(standardDev)
dfstd['Date'] = startH
dfstd['Date'] = pd.to_datetime(dfstd['Date'])
dfstd.columns = ['Standard Deviation', 'Date']

perform(dfrets, dfstd)


# In[8]:


cumRet(monthlyReturn)


# ## Get Weights

# In[9]:


'''
Make one month iteration and return weights vector
Return price vector also

Tom include 2 columns for scaling period and rows 1 and 3 added, and first 4 column titles
Efficiency for storing data to make backtesting quicker
'''

def getWeights(i = len(winners)-1):
    winners = dfList['Winners']
    losers = dfList['Losers']
    startH = dfList['StartH']
    endH = dfList['EndH']

    # Volatility scaling
    # Aiming for 12% annual volatility, which is 3.46% monthly
    targetSD = .0346

    listWin = strip_replace(winners[i])
    listLoss = strip_replace(losers[i])
    numWin = len(listWin)
    p = getCSVPrice(listWin[0], startH[i], endH[i])
    p_l = getCSVPrice(listLoss[0], startH[i], endH[i])

    # Tickers
    retWin = 0
    retLoss = 0
    prices = 10 * np.ones(len(tickers))
    for j in range(numWin):
        p = getCSVPrice(listWin[j], startH[i], endH[i])
        r = getCSVRet(listWin[j], startH[i], endH[i])
        retWin += r/len(listWin)
        k = tickers.index(listWin[j])
        prices[k] = p[len(p)-1]

        p_l = getCSVPrice(listLoss[j], startH[i], endH[i])
        r_l = getCSVRet(listLoss[j], startH[i], endH[i])
        retLoss += r_l/len(listLoss)
        k = tickers.index(listLoss[j])
        prices[k] = p_l[len(p_l)-1]

    # Momentum scaling
    retWinScale = pd.Series()
    retLossScale = pd.Series()
    for j in range(numWin):
        pWin = getCSVPrice(listWin[j], startS[i], endS[i])
        if j == 0:
            retWinScale = 0 * pd.Series(pWin)
            retLossScale = 0 * pd.Series(pWin)
        retWinScale += getCSVRet(listWin[j], startS[i], endS[i])
        pLoss = getCSVPrice(listLoss[j], startS[i], endS[i])
        retLossScale += getCSVRet(listLoss[j], startS[i], endS[i])
    scaleRet = (retWinScale - retLossScale) / len(listWin)
    totalVar = 0
    for j in range(len(scaleRet)):
        retj = float(scaleRet[j])
        totalVar += retj * retj
    monthlyVar = totalVar / 6
    monthlySD = np.sqrt(monthlyVar)
    momSD[i] = monthlySD
    wMo = targetSD / monthlySD

    pSPY = getCSVPrice('SPY', startH[i], endH[i])
    pSPY = pSPY[len(pSPY)-1]
    pVal = getCSVPrice('VTV', startH[i], endH[i])
    pVal = pVal[len(pVal)-1]

    wMo *= .5
    # wHedge = .5
    wHedge = 1 - wMo
    cash = .5-wMo

    # Weights vector
    weights = np.zeros(len(tickers))
    for k in range(numWin):
        j = tickers.index(listWin[k])
        weights[j] = 0.0625 * wMo
    for k in range(numWin):
        j = tickers.index(listLoss[k])
        weights[j] = -0.0625 * wMo

    # Hedge weights
    prices[len(prices)-2] = pVal
    prices[len(prices)-1] = pSPY
    weights[len(weights)-2] = wHedge / 2
    weights[len(weights)-1] = wHedge / 2

    return weights, prices


# ## Fees

# In[10]:


# Making the portfolio in the first month
'''
Volume
0.0035$ per share
0.35$ < fee < 1% of trade value
1% if fee < 0.35$, so if < 100 shares

No sales, so no transaction costs
Unclear as to pass-through fees, assumed to apply to purchases
0.0035*0.000175* number of shares

Clearing
0.0002$ per share

Exchange
0.003$ per quote driven order
-0.002$ per order driver order (a rebate)

'''


#Â Assumed all trades are either quote or order, can make a vector otherwise
#Â Is the short classed as a sale, hence giving transaction costs?
#Â Do we only buy and sell ETFs in whole numbers?
#Â Requires positions as a vector of weights, prices and the stocks in the same order
totalShares = 0
def firstTime(positions, prices=1, quoteDriven=True, liquidity=10000, rebalance=False):
    #Â This may not like the short
    absPos = np.zeros(len(positions))
    for i in range(len(absPos)):
        absPos[i] = positions[i]
        if positions[i] < 0:
            absPos[i] = positions[i] * -1

    if rebalance == False:
        if sum(absPos) != 0 and sum(absPos) != 1:
            sumPos = sum(absPos)
            for i in range(len(positions)):
                 positions[i] /= sumPos
            print('Reweighted: {}'.format(positions))

    if type(prices) == int and prices == 1:
        prices = 10 * np.ones(len(positions))
    else:
        print('Prices: {}'.format(prices))

    numShares = 0
    p = 1
    val = 0
    volume = np.zeros(len(positions))
    trans = np.zeros(len(positions))
    passThru = np.zeros(len(positions))
    clearing = np.zeros(len(positions))
    exchange = np.zeros(len(positions))
    totalShares = 0

    for i in range(len(positions)):
        w = absPos[i]
        p = prices[i]
        val = liquidity * w
        numShares = val / p
        totalShares += numShares
        volume[i] = 0.0035 * numShares
        if volume[i] < 0.35:
            volume[i] = 0.35
        if val / 100 < 0.35:
            volume[i] = val / 100
        if positions[i] < 0:
            trans[i] = 0.0000207 * val
        else:
            trans[i] = 0
        passThru[i] = 0.0000006125 * numShares # 0.0035 * 0.000175
        clearing[i] = 0.002 * numShares
        if absPos[i] > 0:
            if quoteDriven:
                exchange[i] = 0.003
            else:
                exchange[i] = -0.002

    totalVolume = round(sum(volume), 10)
    totalTrans = round(sum(trans), 10)
    totalPass = round(sum(passThru), 10)
    totalClearing = round(sum(clearing), 10)
    totalExchange = round(sum(exchange), 10)
    fees = [totalVolume, totalTrans, totalPass, totalClearing, totalExchange]
    totalFee = round(sum(fees),10)
    print(fees)
    print('\nTotal fee: $ {}'.format(totalFee))
    print('Fee percentage: {} %\n'.format(totalFee/liquidity))

    return totalFee


# ## Rebalance

# In[11]:


# Rebalacing the portfolio
# Needs to have the last month's positions
def rebalance(lastPositions, positions, prices=1, quoteDriven=True, liquidity=10000):
    if type(prices) == int and prices == 1:
        prices = 10 * np.ones(len(positions))

    changePositions = positions - lastPositions

    reFee = firstTime(changePositions, prices, rebalance=True)

    return reFee


# In[12]:


lastMonth = firstTime(getWeights(len(winners)-2)[0], getWeights(len(winners)-2)[1])
lastWeights = getWeights(len(winners)-2)[0]
thisWeights = getWeights()[0]

print('Last month: ')
print(lastMonth)
print('\nThis weights: ')
print(thisWeights)
print('\nLast weights: ')
print(lastWeights)


# ## Exchange Rate

# In[13]:


i = len(winners)-1
listWin = strip_replace(winners[i])
p = getCSVPrice(listWin[0], startH[i], endH[i])
listLoss = strip_replace(losers[i])
p_l = getCSVPrice(listLoss[0], startH[i], endH[i])
numWin = len(listWin)
retLoss = 0
retWin = 0
# Tickers
for j in range(numWin):
    p = getCSVPrice(listWin[j], startH[i], endH[i])
    r = getCSVRet(listWin[j], startH[i], endH[i])
    retWin += r/len(listWin)

    p_l = getCSVPrice(listLoss[j], startH[i], endH[i])
    r_l = getCSVRet(listLoss[j], startH[i], endH[i])
    retLoss += r_l/len(listLoss)

# Momentum strategy return
moRet = retWin - retLoss

# Momentum scaling
pWin = getCSVPrice(listWin[0], startS[i], endS[i])
retWinScale = getCSVRet(listWin[0], startS[i], endS[i])
pLoss = getCSVPrice(listLoss[0], startS[i], endS[i])
retLossScale = getCSVRet(listLoss[0], startS[i], endS[i])
scaleRet = retWinScale - retLossScale
totalVar = 0
for j in range(len(scaleRet)):
    retj = float(scaleRet[j])
    totalVar += retj * retj
monthlyVar = totalVar / 6
monthlySD = np.sqrt(monthlyVar)
wMo = targetSD / monthlySD

pSPY = getCSVPrice('SPY', startH[i], endH[i])
retSPY = getCSVRet('SPY', startH[i], endH[i])
pVal = getCSVPrice('VTV', startH[i], endH[i])
retVal = getCSVRet('VTV', startH[i], endH[i])
hedgeRet = retVal - retSPY

wMo *= .5
# Alternatively wHedge = .5
wHedge = 1 - wMo
cash = .5-wMo

fullPort = wMo * moRet + wHedge * hedgeRet
exRet = get_rate('USD', 'GBP', endH[i]) / get_rate('USD', 'GBP', startH[i])
# fullPort *= exRet
# Make cumulative return and plot it. Try for different weightings
# Want to make a summed monthly return and then add
monthRet = np.sum(fullPort)
afterExRet = monthRet * exRet

print('')
print(afterExRet)
# Should be -0.0020463797297560265


# In[14]:


print(monthRet)
# Before fees and before exchange rate

feee = 0.00037558115138 # %
print(feee)
# Exchange rate return
print(exRet)
# Portfolio return after exchange rate
# print(sum(fullPort))

# Return after exchange rate and fees
print('True return: {}\n'.format((1 + monthRet - feee) * exRet - 1))
# Can only do without fees for full period
# (1 + monthRet) * exRet - 1

# Comment the below after first run
listExRets = np.zeros(len(winners))
for i in range(len(winners)):
    exRet = get_rate('USD', 'GBP', endH[i]) / get_rate('USD', 'GBP', startH[i])
    listExRets[i] = exRet

listExRets = np.array(listExRets)
print(listExRets)


# In[ ]:


print('Change weightings: ')
changeWeights = thisWeights - lastWeights
print(changeWeights)

rebalance(lastWeights, thisWeights, prices)
# $ 3.76 to rebalance in the last month

print(sum(abs(changeWeights)))


# ## Recent Weightings

# In[ ]:


for i in range(len(winners)-2, len(winners)):
    winners = dfList['Winners']
    losers = dfList['Losers']
    startH = dfList['StartH']
    endH = dfList['EndH']

    # Volatility scaling
    # Aiming for 12% annual volatility, which is 3.46% monthly
    targetSD = .0346

    listWin = strip_replace(winners[i])
    listLoss = strip_replace(losers[i])
    p = getCSVPrice(listWin[0], startH[i], endH[i])
    p_l = getCSVPrice(listLoss[0], startH[i], endH[i])
    retWin = 0
    retLoss = 0
    prices = np.ones(len(tickers))
    # Tickers
    for j in range(numWin):
        p = getCSVPrice(listWin[j], startH[i], endH[i])
        r = getCSVRet(listWin[j], startH[i], endH[i])
        retWin += r/len(listWin)
        k = tickers.index(listWin[j])
        prices[k] = p[len(p)-1]

        p_l = getCSVPrice(listLoss[j], startH[i], endH[i])
        r_l = getCSVRet(listLoss[j], startH[i], endH[i])
        retLoss += r_l/len(listLoss)
        k = tickers.index(listLoss[j])
        prices[k] = p_l[len(p_l)-1]

    # Momentum scaling
    pWin = getCSVPrice(listWin[0], startS[i], endS[i])
    retWinScale = getCSVRet(listWin[0], startS[i], endS[i])
    pLoss = getCSVPrice(listLoss[0], startS[i], endS[i])
    retLossScale = getCSVRet(listLoss[0], startS[i], endS[i])
    scaleRet = retWinScale - retLossScale
    totalVar = 0
    for j in range(len(scaleRet)):
        retj = float(scaleRet[j])
        totalVar += retj * retj
    monthlyVar = totalVar / 6
    monthlySD = np.sqrt(monthlyVar)
    wMo = targetSD / monthlySD

    pSPY = getCSVPrice('SPY', startH[i], endH[i])
    pSPY = pSPY[len(pSPY)-1]
    pVal = getCSVPrice('VTV', startH[i], endH[i])
    pVal = pVal[len(pVal)-1]

    wMo *= .5
    # Alternatively wHedge = .5
    wHedge = 1 - wMo
    cash = .5-wMo

    # Weights vector
    weights = np.zeros(len(tickers))
    for k in range(numWin):
        j = tickers.index(listWin[k])
        weights[j] = 0.0625 * wMo
        l = tickers.index(listLoss[k])
        weights[l] = -0.0625 * wMo

    # Hedge weights
    prices[len(prices)-2] = pVal
    prices[len(prices)-1] = pSPY
    weights[len(weights)-2] = 1/4
    weights[len(weights)-1] = -1/4

    if i == len(winners)-2:
        lastWeights = weights
    if i == len(winners)-1:
        thisWeights = weights

print(lastWeights)
print(thisWeights)


# ## wMo

# In[ ]:


m = np.array(momWeight)
l = momSD
m = pd.DataFrame(m).shift(1)
print(m[1:])
print('Mean wMo: ')
print(np.mean(m))
plt.scatter(l, m)


# In[ ]:


# Monthly Sharpe ratio
print(monthlyReturn/standardDev)
print('\nMean Sharpe ratio: {}'.format(np.mean(monthlyReturn/standardDev)))
print('\nSD: {}'.format(np.std(monthlyReturn/standardDev)))


# In[ ]:


print('Momentum monthly return: ')
print(monthlyReturn)
print('\nMomentum SD: ')
print(l)


# In[ ]:


print(momWeight)
print('')
print(np.mean(momWeight))


# In[ ]:


sharpe = (monthlyReturn.mean()*12)/(standardDev.mean()*np.sqrt(252))
print(sharpe)

np.round(monthlyReturn, 10)
# Equivalent to np.round((g_win*momWeight*0.5)-(g_los*momWeight*0.5)+(g_val*hedgeWeight - g_gro*hedgeWeight), 10)


# In[ ]:


frameWinLos = dfList; frameWinLos['Hedge'] = hedgeWeight
frameWinLos = pd.DataFrame.set_index(frameWinLos, frameWinLos['EndH'])
x = pd.DataFrame(momWeight, index=endH); x = x.rename(columns={0:"Momentum_Weights"})
x['Winners_Returns'] = g_win; x['Losers_Returns'] = g_los
x['Hedge_Weight'] = frameWinLos['Hedge']
x['VTV_Return'] = g_val; x['SPY_Return'] = g_spy
x['Winners'] = frameWinLos['Winners']; x['Losers'] = frameWinLos['Losers']
x = pd.DataFrame.sort_index(x, ascending=False)
x['Risk_Metrics'] = 'Risk metrics for Whole Portfolio, Average return: 0.414307%, SD: 0.01645, SemiSD: 0.00956, Max DD: 7.71%, VaR at 5%: 2.314%, CVaR at 5%: 2.695%, Skewness: 0.04543, Excess Kurtosis: -0.543714'
x['Risk_Metrics'][1:] = ''
# x.to_excel("Momentum_with_VTV_SPY_Hedge.xlsx")
y = pd.DataFrame(monthlyReturn, index=endH); y = y.rename(columns={0:"Strategy_Returns"})
x

# Amend end cell for easier reading by text to columns and then transpose 
