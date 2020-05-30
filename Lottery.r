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
data3 <- data2 %>% group_by(Ticker, yrmon) %>% summarise(Return_max=max(Ret, na.rm=TRUE)) %>% drop_na()
data3 <- data3 %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 100))
data3$lagDecile <- lag(data3$decile)
data4 <- data3
cumRet <- data2 %>% group_by(Ticker, yrmon) %>% summarise(MonRet=sum(Ret))
data4 <- merge(data4, cumRet, by=c("Ticker", "yrmon"))
data4 <- data4 %>% group_by(Ticker) %>% arrange(Ticker,yrmon) %>% drop_na()

data4 <- data4 %>% group_by(Ticker) %>% slice(-1) %>% drop_na()
data4 <- data4 %>% group_by(lagDecile, yrmon) %>% summarise(DecRet = mean(MonRet))

plot(data4$yrmon[data4$lagDecile==1], cumsum(100*data4$DecRet[data4$lagDecile==1]), type='l')
for(i in 2:10){
  lines(data4$yrmon[data4$lagDecile==i], cumsum(100*data4$DecRet[data4$lagDecile==i]))
}

# Long worst max and short best max 
firstPort <- data4$DecRet[data4$lagDecile==1] - data4$DecRet[data4$lagDecile==100]
mean(firstPort)
firstPort <- data.frame(unique(data4$yrmon), firstPort)
colnames(firstPort) <- c('yrmon', 'PortRet')
plot(firstPort$yrmon, firstPort$PortRet, type='l')
firstPort$CumRet <- cumsum(firstPort$PortRet)
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

# monthReturn <- monthReturn %>% mutate(yrmon=as.yearmon(monthReturn$StartH))
# combRet <- merge(monthReturn, firstPort, by = "yrmon") %>% arrange(yrmon)
# colnames(combRet) <- c('yrmon', 'date', 'FullMo', 'Lottery', 'Cum')
# combRet <- combRet %>% select(-Cum)
# combRet$CumMo <- cumsum(combRet$FullMo)
# combRet$CumLo <- cumsum(combRet$Lottery)
# 
# par(mfrow=c(2,1))
# plot(combRet$yrmon, combRet$CumMo, type='l')
# lines(combRet$yrmon, combRet$CumLo)
# 
# plot(combRet$yrmon, combRet$CumMo+combRet$CumLo, type='l')
# mean(1200*mean(combRet$FullMo))
# mean(1200*mean(combRet$Lottery))
# mean(1200*mean(combRet$FullMo+combRet$Lottery))




