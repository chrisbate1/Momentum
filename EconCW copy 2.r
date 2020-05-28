rm(list=ls())
setwd('/Users/Chris/Documents/OneDrive - Imperial College London/Fund/Code/')
data <- read.csv('ReturnsSPY.csv'); datacopy <- data
# install.packages(c("plyr", "tidyr", "dplyr", "lubridate", "xts", "PerformanceAnalytics",
#                   "lmtest", "olsrr", "robustbase", "car", "stargazer"))
library(plyr); library(tidyr); library(dplyr); library(lubridate); library(xts);
library(PerformanceAnalytics); library(lmtest); library(olsrr); library(robustbase);
library(car); library(stargazer)

# No cap data here 
data <- datacopy
# data <- data %>% select(-PRC, -CFACPR, -year, -sprtrn, -VOL)
# data <- subset(data, (!is.na(AdjP)|AdjP==0))
data <- data %>% mutate(yrmon=as.yearmon(data$Date))
# data1 <- data %>% group_by(PERMNO) %>% arrange(PERMNO,date)
# data1 <- data %>% group_by(PERMNO, yrmon) %>% summarise(Return_max=max(Ret, na.rm=TRUE)) %>% drop_na()
# data2 <- data %>% group_by(PERMNO, yrmon) %>% filter(row_number()==n())
# medCap <- data %>% group_by(PERMNO, yrmon) %>% summarise(Median_cap=median(cap))
# data2$Median_cap <- medCap$Median_cap
# data3 <- merge(data1, data2, by=c("PERMNO", "yrmon"), all.y=TRUE) %>% arrange(PERMNO, yrmon)
# ME = read.csv('ME_Breakpoints.csv')
# ME$year <- as.character(ME$Date) %>% substr(.,1,4)
# ME$month <- as.character(ME$Date) %>% substr(.,5,6)
# ME$yrmon <- as.yearmon(as.Date(paste(ME$year, ME$month, "01", sep='-')))
# ME <- ME %>% select(., 'P50', yrmon)
# ME$P50 <- ME$P50*1000000
# ME$yrmon <- ME$yrmon
# data4 <- merge(data3, ME, by="yrmon") %>% arrange(PERMNO, yrmon)
# TOP_ME <- subset(data4, cap>=P50)
# BOT_ME <- subset(data4, cap<P50)
# TOP_ME <- TOP_ME %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 10))
# BOT_ME <- BOT_ME %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 10))
# TOP_ME <- TOP_ME %>% group_by(decile, yrmon) %>% summarise(n=n()) %>% group_by(decile) %>% summarise(avg_n=mean(n))
# BOT_ME <- BOT_ME %>% group_by(decile, yrmon) %>% summarise(n=n()) %>% group_by(decile) %>% summarise(avg_n=mean(n))
# 
# DECILES <- cbind(BOT_ME, TOP_ME[,2])
# colnames(DECILES) <- c('Decile', 'Bottom', 'Top')

data2 <- data %>% gather(yrmon, Ticker, NLR:SPY) %>% ungroup()
colnames(data2) <- c('Date', 'Ticker', 'Ret')
data2 <- data2 %>% mutate(yrmon=as.yearmon(data2$Date))
data3 <- data2 %>% group_by(Ticker, yrmon) %>% summarise(Return_max=max(Ret, na.rm=TRUE)) %>% drop_na()
data3 <- data3 %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 10))
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
firstPort <- data4$DecRet[data4$lagDecile==1] - data4$DecRet[data4$lagDecile==10]
mean(firstPort)
firstPort <- data.frame(unique(data4$yrmon), firstPort)
colnames(firstPort) <- c('yrmon', 'PortRet')
plot(firstPort$yrmon, firstPort$PortRet, type='l')
firstPort$CumRet <- cumsum(firstPort$PortRet)
plot(firstPort$yrmon, firstPort$CumRet, type='l')
# Mean annual return 
mean(1200*firstPort$PortRet)

plot(data4$yrmon[data4$lagDecile==1], cumsum(data4$DecRet[data4$lagDecile==1]), type='l', ylim=c(-1, 1/2))
lines(data4$yrmon[data4$lagDecile==10], cumsum(data4$DecRet[data4$lagDecile==10]))

plot(firstPort$yrmon, firstPort$CumRet, type='l')

monthReturn <- read.csv('FullRet.csv')
monthReturn <- monthReturn %>% mutate(yrmon=as.yearmon(monthReturn$StartH))
combRet <- merge(monthReturn, firstPort, by = "yrmon") %>% arrange(yrmon)
colnames(combRet) <- c('yrmon', 'date', 'FullMo', 'Lottery', 'Cum')
combRet <- combRet %>% select(-Cum)
combRet$CumMo <- cumsum(combRet$FullMo)
combRet$CumLo <- cumsum(combRet$Lottery)

par(mfrow=c(2,1))
plot(combRet$yrmon, combRet$CumMo, type='l')
lines(combRet$yrmon, combRet$CumLo)

plot(combRet$yrmon, combRet$CumMo+combRet$CumLo, type='l')
mean(1200*mean(combRet$FullMo))
mean(1200*mean(combRet$Lottery))
mean(1200*mean(combRet$FullMo+combRet$Lottery))

write.csv(combRet, 'CombRet.csv', row.names =FALSE)

# # TOP_ME <- subset(data4, Median_cap>=P50)
# # BOT_ME <- subset(data4, Median_cap<P50)
# # TOP_ME <- TOP_ME %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 10))
# # BOT_ME <- BOT_ME %>% group_by(yrmon) %>% mutate(decile=ntile(Return_max, 10))
# 
# # TOP Portfolio
# TOP_ME$lagDecile <- lag(TOP_ME$decile)
# TOP_ME <- TOP_ME %>% group_by(PERMNO) %>% slice(-1) %>% drop_na() %>% filter(lagDecile == 1 | lagDecile == 10)
# TOP_ME$MonRet[TOP_ME$lagDecile == 1] <- TOP_ME$MonRet[TOP_ME$lagDecile == 1] * -1
# TOP_PORT <- TOP_ME %>% group_by(lagDecile, yrmon) %>% summarise(PosRet = mean(MonRet))
# TOP_PORT <- TOP_PORT %>% group_by(yrmon) %>% summarise(PortRet = sum(PosRet))
# TOP_PORT$CumRet <- cumsum(TOP_PORT$PortRet)
# topMonthRet <- mean(TOP_PORT$PortRet)
# topYearRet <- 12 * topMonthRet
# topTotRet <- 25 * topYearRet
# 
# # BOT Portfolio
# BOT_ME$lagDecile <- lag(BOT_ME$decile)
# BOT_ME <- BOT_ME %>% group_by(PERMNO) %>% slice(-1) %>% drop_na() %>% filter(lagDecile == 1 | lagDecile == 10)
# BOT_ME$MonRet[BOT_ME$lagDecile == 1] <- BOT_ME$MonRet[BOT_ME$lagDecile == 1] * -1
# BOT_PORT <- BOT_ME %>% group_by(lagDecile, yrmon) %>% summarise(PosRet = mean(MonRet))
# BOT_PORT <- BOT_PORT %>% group_by(yrmon) %>% summarise(PortRet = sum(PosRet))
# BOT_PORT$CumRet <- cumsum(BOT_PORT$PortRet)
# botMonthRet <- mean(BOT_PORT$PortRet)
# botYearRet <- 12 * botMonthRet
# botTotRet <- 25 * botYearRet
# 
# # Summary Stats
# options(warn=-1)
# plot(TOP_PORT[,1:2], main="TOP PORTFOLIO", type="l")
# plot(BOT_PORT[,1:2], main="BOT PORTFOLIO", type="l")
# summary(TOP_PORT$PortRet); summary(BOT_PORT$PortRet)
# cat("TOP SD: ", sd(sqrt(12)*TOP_PORT$PortRet), "\n"); cat("BOT SD: ",sd(sqrt(12)*BOT_PORT$PortRet), "\n")
# cat("TOP t: ", mean(TOP_PORT$PortRet)/sd(TOP_PORT$PortRet), "\n"); cat("BOT t: ", mean(BOT_PORT$PortRet)/sd(BOT_PORT$PortRet), "\n")
# cat("TOP Sharpe: ", sqrt(12)*mean(TOP_PORT$PortRet)/sd(TOP_PORT$PortRet), "\n"); cat("BOT Sharpe: ", sqrt(12)*mean(BOT_PORT$PortRet)/sd(BOT_PORT$PortRet), "\n")
# cat("TOP Inv Sharpe: ", -sqrt(12)*mean(TOP_PORT$PortRet)/sd(TOP_PORT$PortRet), "\n"); cat("BOT Inv Sharpe: ", -sqrt(12)*mean(BOT_PORT$PortRet)/sd(BOT_PORT$PortRet), "\n")
# cat("TOP Max Drawdown: ", maxDrawdown(TOP_PORT$PortRet), "\n"); cat("BOT Max Drawdown: ", maxDrawdown(BOT_PORT$PortRet), "\n")
# cat("Inverse TOP Max Drawdown: ", maxDrawdown(-TOP_PORT$PortRet), "\n"); cat("Inverse BOT Max Drawdown: ", maxDrawdown(-BOT_PORT$PortRet), "\n")
# plot(TOP_PORT[,c(1, 3)], main="TOP CUMULATIVE", type="l")
# plot(BOT_PORT[,c(1, 3)], main="BOT CUMULATIVE", type="l")
# options(warn=0)
# 
# plot(BOT_PORT[,c(1, 3)], main="Cumulative Returns", type="l", xlab="Year", ylab="Portfolio Return", col="red", ylim=range(5,-15))
# lines(TOP_PORT[,c(1, 3)], type="l", col="blue")
# legend("bottomleft", legend=c("TOP", "BOT"), cex=.8, inset=c(0.05,0.05), col=c("blue", "red"), lty=1:1)
# 
# # Fama-French Factors
# FF = read.csv('FF5.csv')
# FF$yrmon <- as.yearmon(as.Date(paste(as.character(FF$Date) %>% substr(.,1,4), as.character(FF$Date) %>% substr(.,5,6), "01", sep='-')))
# FF <- FF %>% select(-Date) %>% arrange(yrmon)
# FF[1:6] <- FF[1:6] / 100
# FFPorts <- merge(FF, TOP_PORT[,1:2], by="yrmon") %>% arrange(yrmon)
# FFPorts <- merge(FFPorts, BOT_PORT[,1:2], by="yrmon") %>% arrange(yrmon)
# colnames(FFPorts)[2] <- "MktRF"; colnames(FFPorts)[8] <- "TopRet"; colnames(FFPorts)[9] <- "BotRet"
# FFPorts <- FFPorts %>% drop_na()
# 
# # Regression
# topModel <- lm(TopRet~MktRF+SMB+HML+RMW+CMA, data=FFPorts)
# botModel <- lm(BotRet~MktRF+SMB+HML+RMW+CMA, data=FFPorts)
# stargazer(topModel)
# stargazer(botModel)
# topSum <- summary(topModel)
# botSum <- summary(botModel)
# # Residual plots
# plot(density(resid(topModel)))
# xfit <- seq(min(resid(topModel)),max(resid(topModel)),length=40)
# yfit <- dnorm(xfit,mean=mean(resid(topModel)),sd=sd(resid(topModel)))
# lines(xfit, yfit, col="blue")
# plot(density(resid(botModel)))
# xfit <- seq(min(resid(botModel)),max(resid(botModel)),length=40)
# yfit <- dnorm(xfit,mean=mean(resid(botModel)),sd=sd(resid(botModel)))
# lines(xfit, yfit, col="blue")
# # Sharpe ratios
# topSR <- sqrt(12)*mean((12*FFPorts$TopRet-FFPorts$RF)/sd(12*FFPorts$TopRet-FFPorts$RF))
# botSR <- sqrt(12)*mean((12*FFPorts$BotRet-FFPorts$RF)/sd(12*FFPorts$BotRet-FFPorts$RF))
# topSRInv <- sqrt(12)*mean((-12*FFPorts$TopRet-FFPorts$RF)/sd(-12*FFPorts$TopRet-FFPorts$RF))
# botSRInv <- sqrt(12)*mean((-12*FFPorts$BotRet-FFPorts$RF)/sd(-12*FFPorts$BotRet-FFPorts$RF))
# # Functional form
# topReset <- resettest(topModel, power=2:3, type="regressor", data=FFPorts)
# botReset <- resettest(botModel, power=2:3, type="regressor", data=FFPorts)
# # Heteroscedasticity
# topBP <- bptest(topModel)
# botBP <- bptest(botModel)
# # Autocorrelation
# print("Test statistics: [1.667, 1.748, 2.252, 2.333]")
# topDW <- dwtest(topModel)
# botDW <- dwtest(botModel)
# # Residual normality
# topShap <- shapiro.test(topModel$residuals)
# botShap <- shapiro.test(botModel$residuals)
# # Multicollinearity
# corMat = cor(FFPorts[,2:6])
# vifs <- vif(topModel)
# # Robust standard errors
# robTopModel <- lmrob(TopRet~MktRF+SMB+HML+RMW+CMA, data=FFPorts)
# robBotModel <- lmrob(BotRet~MktRF+SMB+HML+RMW+CMA, data=FFPorts)
# stargazer(robTopModel)
# stargazer(robBotModel)
# robTopSum <- summary(robTopModel)
# robBotSum <- summary(robBotModel)
# # Value at risk 5%  
# topVaR <- VaR(FFPorts$TopRet)[,1]
# botVaR <- VaR(FFPorts$BotRet)[,1]
