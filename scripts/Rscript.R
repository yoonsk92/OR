library('ggplot2')
library('forecast')
library('tseries')
library('xlsx')
getwd()
setwd("C:/Users/yoons/Desktop/York/OMIS4000 Models and Applications of OR/Project/Daily Orders")
aa <- as.numeric(unlist(train[,2]))
y <- msts(aa, seasonal.periods=c(7,30.4,365.25))
fit <- tbats(y)
fc <- forecast(fit)
plot(fc)
fc$model
orderpred1 <- fc$mean
View(orderpred1)
components <- tbats.components(fit)
plot(components)
checkresiduals(fit)
accuracy(fc)
quartile <- cbind(fc$lower, fc$upper)
View(quartile)
fc$model
fc$level
write.csv(orderpred1, "orderpred.csv")

