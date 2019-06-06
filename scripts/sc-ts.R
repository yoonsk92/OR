library(ggplot2)
library(ggthemes)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(ggfortify)
library(ggrepel)
library(RColorBrewer)
library(data.table)
library(dplyr)
library(readr)
library(tibble)
library(tidyr)
library(lazyeval)
library(broom)
library(stringr)
library(purrr)
library(forcats)
library(lubridate)
library(forecast)
library(prophet)

df <- read.csv("Spa_ServiceSales_All.csv", header = T, sep = ",", stringsAsFactors = F)
df$a <- NA
df$b <- NA
df$c <- NA

#colname & remove first row & clean columns
names(df) <- c("Order Date", "Order Number", "Employee Name", "Customer First Name", "Customer Last Name", "Category", "Sub Category", "Service Name", "Qty", "Amount", "Adjustment", "Total Sales", "Tax", "Refunds")
df <- df[-1,]
df[,2:14] <- NA

dates_ <- grep('\\d+\\/\\d+\\/\\d+', df$`Order Date`, value = T)

#Time Series Analysis for Services
df <- read.csv("Spa_ServiceSales_All.csv", header = T, sep = ",", stringsAsFactors = F)
