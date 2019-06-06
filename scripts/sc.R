library(xts)

#Seasonality in Revenue
getwd()
setwd("C:/Users/yoons/Desktop/York/OMIS4000 Models and Applications of OR/Project")

rev <- read.csv("Revenue.csv", header = T, sep = ",", stringsAsFactors = FALSE)
str(rev)

rev$Revenue <- as.numeric(gsub('[$,.]','',rev$Revenue))
rev_ts <- ts(rev$Revenue, start = c(2015,2), end = c(2018,2), freq = 12)

frequency(rev_ts)
cycle(rev_ts)
summary(rev_ts)

plot(rev_ts, xlab = "Date", ylab = "Revenue")
plot(as.xts(rev_ts), major.format = "%Y-%m", main = "", ylab = "Revenue")
options(scipen=1)

dec_rev_ts <- decompose(rev_ts, "multiplicative")
plot(dec_rev_ts)



