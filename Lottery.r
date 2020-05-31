rm(list=ls())
setwd('/Users/Chris/Downloads/')
data <- read.csv('gbrets.csv'); datacopy <- data
# install.packages(c("plyr", "tidyr", "dplyr", "lubridate", "xts", "PerformanceAnalytics",
#                   "lmtest", "olsrr", "robustbase", "car", "stargazer"))
library(plyr); library(tidyr); library(dplyr); library(lubridate); library(xts);
library(PerformanceAnalytics); library(lmtest); library(olsrr); library(robustbase);
library(car); library(stargazer)

# No cap data here
data <- datacopy
data <- data %>% mutate(yrmon=as.yearmon(data$date))

data[is.na(data)] <- 0
data2 <- data %>% gather(yrmon, Ticker, X1932:X338444) %>% ungroup()
colnames(data2) <- c('Date', 'Ticker', 'Ret')
data2 <- data2 %>% mutate(yrmon=as.yearmon(data2$Date))
data3 <- data2 %>% group_by(Ticker, yrmon) %>% summarise(Return_max=max(Ret, na.rm=TRUE), Return_min=min(Ret, na.rm=TRUE)) %>% drop_na()
data3 <- data3 %>% group_by(yrmon) %>% mutate(decileMax=ntile(Return_max, 100), decileMin=ntile(Return_min, 100))
data3$lagDecileMax <- lag(data3$decileMax)
data3$lagDecileMin <- lag(data3$decileMin)
data4 <- data3
cumRet <- data2 %>% group_by(Ticker, yrmon) %>% summarise(MonRet=sum(Ret))
data4 <- merge(data4, cumRet, by=c("Ticker", "yrmon"))
data4 <- data4 %>% group_by(Ticker) %>% arrange(Ticker,yrmon) %>% drop_na()
data4 <- data4 %>% group_by(Ticker) %>% slice(-1) %>% drop_na()

maxData4 <- data4 %>% group_by(yrmon, lagDecileMax) %>% summarise(DecRet = mean(MonRet))
minData4 <- data4 %>% group_by(yrmon, lagDecileMin) %>% summarise(DecRet = mean(MonRet))

plot(maxData4$yrmon[maxData4$lagDecileMax==100], cumsum(-maxData4$DecRet[maxData4$lagDecileMax==100]), type='l')
for(i in 1:10){
  lines(maxData4$yrmon[maxData4$lagDecileMax==(100-i)], cumsum(-maxData4$DecRet[maxData4$lagDecileMax==(100-i)]))
}

plot(minData4$yrmon[minData4$lagDecileMin==1], cumsum(-minData4$DecRet[minData4$lagDecileMin==1]), type='l')
for(i in 2:10){
  lines(minData4$yrmon[minData4$lagDecileMin==i], cumsum(-minData4$DecRet[minData4$lagDecileMin==i]))
}

# Long worst max 1 and short best max 100
# Long best min 100 and short worst min 1
firstPort <- maxData4[maxData4$lagDecileMax==1,] %>% select(-DecRet)
firstPort$lagDecileMax <- maxData4$DecRet[maxData4$lagDecileMax==1] - maxData4$DecRet[maxData4$lagDecileMax==100]
firstPort$MinRet <- minData4$DecRet[minData4$lagDecileMin==100] - minData4$DecRet[minData4$lagDecileMin==1]
colnames(firstPort) <- c('yrmon', 'MaxRet', 'MinRet')
firstPort$PortRet <- firstPort$MaxRet + firstPort$MinRet
firstPort$CumRet <- cumsum(firstPort$PortRet)
plot(firstPort$yrmon, firstPort$CumRet, type='l')

firstPort$MinDecRet <- data4$MinDecRet[data4$lagDecileMin==100] - data4$MinDecRet[data4$lagDecileMin==1]
mean(firstPort)
firstPort <- data.frame(unique(data4$yrmon), firstPort)
colnames(firstPort) <- c('yrmon', 'MaxPortRet', 'MinPortRet')
plot(firstPort$yrmon, firstPort$MaxPortRet+firstPort$MinPortRet, type='l')
firstPort$CumMaxRet <- cumsum(firstPort$CumMaxRet)
plot(firstPort$yrmon, firstPort$CumRet, type='l')
# Mean annual return
mean(1200*firstPort$PortRet)

plot(data4$yrmon[data4$lagDecile==100], cumsum(data4$DecRet[data4$lagDecile==100]), type='l')
lines(data4$yrmon[data4$lagDecile==1], cumsum(data4$DecRet[data4$lagDecile==1]))

plot(firstPort$yrmon, log(1+firstPort$CumRet, base = 10), type='l')
plot(firstPort$yrmon, firstPort$CumRet, type='l')

thisMonth <- data3[data3$yrmon == max(data3$yrmon), ]
thisMonth$Ticker <- as.numeric(substring(thisMonth$Ticker, 2))
thisMonth <- thisMonth[thisMonth$lagDecile == 100, ]
plot(data4$yrmon[data4$lagDecile==100], -cumsum(data4$DecRet[data4$lagDecile==100]), type='l')

compInfo <- read.csv('compInfo.csv')[,-1]
thisMonth <- merge(thisMonth, compInfo, by.x = "Ticker", by.y = "gvkey")

lotteryMax <- firstPort
plot(lotteryMax$yrmon, lotteryMax$CumRet+lotteryMin$CumRet, type='l')
plot(lotteryMax$yrmon, lotteryMax$CumRet-lotteryMin$CumRet, type='l')
lottery <- lotteryMax
lottery[,2:3] <- lotteryMax[,2:3] + lotteryMin[,2:3]
plot(lottery$yrmon, lottery$CumRet, type='l')
