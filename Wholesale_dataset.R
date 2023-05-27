#The analysis is performed on the “Wholesale customers” dataset from the UCI Machine Learning Repository[http://archive.ics.uci.edu/ml]. 
#The data set refers to clients of a wholesale distributor. 
#It includes the annual spending in monetary units (m.u.) on diverse product categories
#The target feature are:
#•FRESH: annual spending (m.u.) on fresh products (Continuous);
#•MILK: annual spending (m.u.) on milk products (Continuous);
#•GROCERY: annual spending (m.u.)on grocery products (Continuous);
#•FROZEN: annual spending (m.u.)on frozen products (Continuous);
#•DETERGENTS PAPER: annual spending (m.u.) on detergents and paper products (Continuous);
#•DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
#•CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafè) or Retail channel (Nominal);
#•REGION: customers Region are splitted in: Lisbon, Oporto or Other Region (Nominal)


library("readr")
library(moments)
library("gamlss")
library(MASS)
library("fitdistrplus")
library("vcd")
library("corrplot")
library(gridExtra)
library(lmtest)
library("pca3d")
library("ggplot2")
library("factoextra")
library(cluster)
library("hopkins")
library("NbClust")
library("fpc")
library("mclust")
library("clValid")
library("gamlss.mx")

##Feature Analysis
data <- read_csv("C:/Users/acer/Downloads/Wholesale customers data.csv")
str(data)
head(data)
dim(data)

data$Channel[data$Channel==1]<-"Ho.re.ca"
data$Channel[data$Channel==2]<-"Retail"
data$Region[data$Region==1]<-"Lisbon"
data$Region[data$Region==2]<-"Oporto"
data$Region[data$Region==3]<-"Other"
head(data)

data$Channel<-as.factor(data$Channel)
data$Region<-as.factor(data$Region)
summary(data)

##About Channel
#is a nominal categorical variable that represents two different Channel of distribution
fac_channel<-as.factor(data$Channel)
summary(fac_channel)
pie(table(data$Channel), labels =paste0( round((table(data$Channel)/length(data$Channel))*100,2) , "%") ,main = "CHANNEL",col = rainbow(length(table(data$Channel))))
legend("topright", names(table(data$Channel)), cex = 0.8,
       fill = rainbow(length(table(data$Channel))))

##About Region
#is a nominal categorical variable that represents 3 most important Region for this study
fac_region<-as.factor(data$Region)
summary(fac_region)
pie(table(data$Region), labels =paste0( round((table(data$Region)/length(data$Region))*100,2) , "%") ,main = "REGION",col = rainbow(length(table(data$Region))))
legend("topright", names(table(data$Region)), cex = 0.8,
       fill = rainbow(length(table(data$Region))))

##About Annual Spending on Fresh Product
#is a numerical continuous variable, defined on the interval [0,+∞)
summary(data$Fresh)

kurtosis(data$Fresh)
#A positive kurtosis value indicates we are dealing with a fat tailed distribution, where extreme outcomes are more common than would be predicted by a standard normal distribution. 
#if the kurtosis = 3.0 is a mesokurtic distribution. 
#This distribution has a kurtosis similar to that of the normal distribution, meaning the extreme value characteristic of the distribution is similar to that of a normal distribution.
#In this case, k>3 so we have a leptokurtic distribution. Any distribution that is leptokurtic displays greater kurtosis than a mesokurtic distribution. 
#This distribution appears as a curve one with long tails (outliers)
#The final type of distribution is platykurtic distribution. These types of distributions have short tails (fewer outliers). 
#Platykurtic distributions have demonstrated more stability than other curves

skewness(data$Fresh)
#A positive skewness would indicate a distribution would be biased towards higher values, such that the mean of the distribution will exceed the median of the distribution. 
#Right Skewed distributions greater than 0.8 often indicate the presence of a handful of exceptionally high outliers.

jarque.test(data$Fresh)
#For this test the Null Hypothesis is: the dataset has a skewness and kurtosis that matches a normal distribution.
#While Alternative Hypothesis is: the dataset has a skewness and kurtosis that does not match a normal distribution.

box<-boxplot(data$Fresh)
#The boxplot shows us the presence of many outliers
box$out
data[which(data$Fresh%in% box$out),]
#Most outliers come from Channel Ho.re.ca, this might be related to the fact that 
#Ho.re.ca is characterized by high values of annual spending on Fresh Product than other channel.

table(data$Fresh)
length(unique(data$Fresh))
quantile(data$Fresh, 0.25)
quantile(data$Fresh, 0.50)
quantile(data$Fresh, 0.75)

hist(data$Fresh, main="Annual Spending on Fresh Product", ylim=c(0,6e-05), col="green", freq = FALSE)
lines(density(data$Fresh), col="red", lwd=2)

##Model Fitting of Fresh
#According to the domain of the variable, different distributions are fitted on the data. 
#SBC, AIC and Loglikelihood value are used to evaluate the best fitted model.
par(mfrow=c(3,2))
Fresh.Exp<-histDist(data$Fresh, family=EXP, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Exponential Distributed")
Fresh.Ga<-histDist(data$Fresh, family=GA, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Gamma Distributed")
Fresh.Gg<-histDist(data$Fresh, family=GG, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Generalized Gamma Distributed")
Fresh.Ig<-histDist(data$Fresh, family=IG, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Invers Gaussian Distributed")
Fresh.Logn<-histDist(data$Fresh, family=LOGNO, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Log-Normal Distributed")
Fresh.Wei<-histDist(data$Fresh, family=WEI, xlab="Annual Spending on Fresh Product", nbins=50, main="Spending on Fresh Product_Weibull Distributed")

data.frame(row.names=c("Exponential","Gamma","Generalized Gamma","Inverse Gaussian","Log-normal","Weibull"),
           df=c(Fresh.Exp$df.fit,Fresh.Ga$df.fit,Fresh.Gg$df.fit,Fresh.Ig$df.fit,Fresh.Logn$df.fit,Fresh.Wei$df.fit),
           AIC=c(AIC(Fresh.Exp),AIC(Fresh.Ga),AIC(Fresh.Gg),AIC(Fresh.Ig),AIC(Fresh.Logn),AIC(Fresh.Wei)),
           SBC=c(Fresh.Exp$sbc,Fresh.Ga$sbc,Fresh.Gg$sbc,Fresh.Ig$sbc,Fresh.Logn$sbc,Fresh.Wei$sbc),
           LogLik=c(logLik(Fresh.Exp),logLik(Fresh.Ga),logLik(Fresh.Gg),logLik(Fresh.Ig),logLik(Fresh.Logn),logLik(Fresh.Wei)))
#According to this dataframe the best Distribution that we can obtain from the AIC criterion is the "Gamma Distribution", 
#while for the SBC criterion is the "Exponential Distribution, and for the maximization of the LogLik is the "Generilize Gamma Distribution".
#But it's possible compare other model in order to find the model which best fit our data using the function:
fitDist(data$Fresh,type="realplus")
#So, according to "fitDist" function, the best model is "Box-Cox-t-orig."
par(mfrow= c(1,1))
Fresh.BCTo<-histDist(data$Fresh, family=BCTo, xlab="Annual Spending on Fresh Product", nbins=50, main="Annual Spending on Fresh Product_Box-Cox-t-orig. Distributed")
data.frame(row.names="Box-Cox-t-orig.", df=Fresh.BCTo$df.fit, AIC=AIC(Fresh.BCTo),
           SBC=Fresh.BCTo$sbc, LogLik=logLik(Fresh.BCTo))
#The "BCTo" distribution minimize the AIC while maximize the LogLik, but the best outcome from SBC is obtained by the Exponential Distribution



##About Annual Spending on Milk Product
#is a numerical continuous variable, defined on the interval [0,+∞)
summary(data$Milk)
kurtosis(data$Milk)
#Since the kurtosis is greater than 3, this indicates that the distribution has more values in the tails compared to a normal distribution.
#The "skinniness" of a leptokurtic distribution is a consequence of the outliers, which stretch the horizontal axis of the histogram graph, 
#making the bulk of the data appear in a narrow ("skinny") vertical range.
skewness(data$Milk)
#Since the skewness is positive, this indicates that the distribution is right-skewed. 
#This confirms what we have seen in the histogram.
par(mfrow= c(1,1))
jarque.test(data$Milk)
box<-boxplot(data$Milk)
box$out
data[which(data$Milk%in% box$out),]
#Most outliers come from Channel Retail, 
#this might be related to the fact that the Retail is characterized by high values 
#of annual spending score on Milk Product than the other Channel.
table(data$Milk)
length(unique(data$Milk))
quantile(data$Milk, 0.25)
quantile(data$Milk, 0.50)
quantile(data$Milk, 0.75)

par(mfrow= c(1,1))
hist(data$Milk, main="Annual Spending on Milk Product", ylim=c(0,1.5e-04), col="darkmagenta", freq = FALSE)
lines(density(data$Milk), col="red")


#Model Fitting of Milk Product
par(mfrow=c(3,2))
Milk.Exp<-histDist(data$Milk, family=EXP, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Exponential Distributed")
Milk.Ga<-histDist(data$Milk, family=GA, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Gamma Distributed")
Milk.Gg<-histDist(data$Milk, family=GG, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Generalized Gamma Distributed")
Milk.Ig<-histDist(data$Milk, family=IG, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Invers Gaussian Distributed")
Milk.Logn<-histDist(data$Milk, family=LOGNO, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Log-Normal Distributed")
Milk.Wei<-histDist(data$Milk, family=WEI, xlab="Annual Spending on Milk Product", nbins=50, main="Spending on Milk Product_Weibull Distributed")

data.frame(row.names=c("Exponential","Gamma","Generalized Gamma","Inverse Gaussian","Log-normal","Weibull"),
           df=c(Milk.Exp$df.fit,Milk.Ga$df.fit,Milk.Gg$df.fit,Milk.Ig$df.fit,Milk.Logn$df.fit,Milk.Wei$df.fit),
           AIC=c(AIC(Milk.Exp),AIC(Milk.Ga),AIC(Milk.Gg),AIC(Milk.Ig),AIC(Milk.Logn),AIC(Milk.Wei)),
           SBC=c(Milk.Exp$sbc,Milk.Ga$sbc,Milk.Gg$sbc,Milk.Ig$sbc,Milk.Logn$sbc,Milk.Wei$sbc),
           LogLik=c(logLik(Milk.Exp),logLik(Milk.Ga),logLik(Milk.Gg),logLik(Milk.Ig),logLik(Milk.Logn),logLik(Milk.Wei)))

#According to the AIC and the LogLik criterions the best model is the "Generalized Gamma Distribution", 
#while for the SBC criterion the best model is Log-normal.

#But it's possible compare other model in order to find the model which best fit our data using the function:
fitDist(data$Milk,type="realplus")
#So, according to "fitDist" function, the best model is "Box-Cox-Cole-Green"
par(mfrow= c(1,1))
Milk.BCCG<-histDist(data$Milk, family=BCCG, xlab="Annual Spending on Milk Product", nbins=50, main="Annual Spending on Milk Product_Box-Cox-Cole-Green Distributed")
data.frame(row.names="Box-Cox-Cole-Green", df=Milk.BCCG$df.fit, AIC=AIC(Milk.BCCG),
           SBC=Milk.BCCG$sbc, LogLik=logLik(Milk.BCCG))
#The "BCCG" distribution minimize the AIC while maximize the LogLik, 
#but the best outcome according to SBC criterion is obtained by the Log-normal Distribution


##About Annual Spending on Grocery Product
#is a numerical continuous variable, defined on the interval [0,+∞)
summary(data$Grocery)
kurtosis(data$Grocery)
skewness(data$Grocery)
jarque.test(data$Grocery)

par(mfrow= c(1,1))
box<-boxplot(data$Grocery)
box$out
data[which(data$Grocery%in% box$out),]
#In this case all the outliers come from the Retail channel, 
#this might be releated that the Retail is characterized by high values of annual spending score on Grocery Product than the other Channel.
table(data$Grocery)
length(unique(data$Grocery))
quantile(data$Grocery, 0.25)
quantile(data$Grocery, 0.50)
quantile(data$Grocery, 0.75)

par(mfrow= c(1,1))
hist(data$Grocery, main="Annual Spending on Grocery Product", ylim=c(0,1e-04), col="orange", freq = FALSE)
lines(density(data$Grocery), col="black")


#Model Fitting of Grocery Product
par(mfrow=c(3,2))
Grocery.Exp<-histDist(data$Grocery, family=EXP, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Exponential Distributed")
Grocery.Ga<-histDist(data$Grocery, family=GA, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Gamma Distributed")
Grocery.Gg<-histDist(data$Grocery, family=GG, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Generalized Gamma Distributed")
Grocery.Ig<-histDist(data$Grocery, family=IG, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Invers Gaussian Distributed")
Grocery.Logn<-histDist(data$Grocery, family=LOGNO, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Log-Normal Distributed")
Grocery.Wei<-histDist(data$Grocery, family=WEI, xlab="Annual Spending on Grocery Product", nbins=50, main="Spending on Grocery Product_Weibull Distributed")

data.frame(row.names=c("Exponential","Gamma","Generalized Gamma","Inverse Gaussian","Log-normal","Weibull"),
           df=c(Grocery.Exp$df.fit,Grocery.Ga$df.fit,Grocery.Gg$df.fit,Grocery.Ig$df.fit,Grocery.Logn$df.fit,Grocery.Wei$df.fit),
           AIC=c(AIC(Grocery.Exp),AIC(Grocery.Ga),AIC(Grocery.Gg),AIC(Grocery.Ig),AIC(Grocery.Logn),AIC(Grocery.Wei)),
           SBC=c(Grocery.Exp$sbc,Grocery.Ga$sbc,Grocery.Gg$sbc,Grocery.Ig$sbc,Grocery.Logn$sbc,Grocery.Wei$sbc),
           LogLik=c(logLik(Grocery.Exp),logLik(Grocery.Ga),logLik(Grocery.Gg),logLik(Grocery.Ig),logLik(Grocery.Logn),logLik(Grocery.Wei)))
#According to the criterions the best model is the "Generalized Gamma".
#But it's possible compare other model in order to find the model which best fit our data using the function:
fitDist(data$Grocery, type="realplus")
#So according to the function "fitDist" the best model is "Box-Cox Power Exponential-orig."

par(mfrow= c(1,1))
Grocery.BCPEo<-histDist(data$Grocery, family=BCPEo, xlab="Annual Spending on Grocery Product", nbins=50, main="Annual Spending on Grocery Product_Box-Cox Power Exponential-orig. Distributed")
data.frame(row.names="Box-Cox Power Exponential-orig.", df=Grocery.BCPEo$df.fit, AIC=AIC(Grocery.BCPEo),
           SBC=Grocery.BCPEo$sbc, LogLik=logLik(Grocery.BCPEo))
#So according to the all criterions and the "fitDist" function: 
#the best model is the "Box-Cox Power Exponential-orig."


##About Annual Spending on Frozen Product
#is a numerical continuous variable, defined on the interval [0,+∞)
summary(data$Frozen)
kurtosis(data$Frozen)
#Since the kurtosis is greater than 3, this indicates that the distribution has more values in the tails compared to a normal distribution.
#The "skinniness" of a leptokurtic distribution is a consequence of the outliers, which stretch the horizontal axis of the histogram graph, 
#making the bulk of the data appear in a narrow ("skinny") vertical range.

skewness(data$Frozen)
#A positive skewness would indicate a distribution would be biased towards higher values, such that the mean of the distribution will 
#exceed the median of the distribution. Right Skewed distributions greater than 0.8 often indicate the presence of a handful of exceptionally high outliers.

jarque.test(data$Frozen)
#For this test the Null Hypothesis is: the dataset has a skewness and kurtosis that matches a normal distribution.
#While Alternative Hypothesis is: the dataset has a skewness and kurtosis that does not match a normal distribution.

par(mfrow= c(1,1))
box<-boxplot(data$Frozen)
box$out
data[which(data$Frozen%in% box$out),]
#Most outliers come from Channel Ho.re.ca, this might be related to the fact that 
#Ho.re.ca is characterized by high values of annual spending on Frozen Product than other channel.

table(data$Frozen)
length(unique(data$Frozen))
quantile(data$Frozen, 0.25)
quantile(data$Frozen, 0.50)
quantile(data$Frozen, 0.75)

par(mfrow= c(1,1))
hist(data$Frozen, main="Annual Spending on Frozen Product", ylim=c(0,3e-04), col="yellow", freq = FALSE)
lines(density(data$Frozen), col="black")


#Model Fitting of Frozen Product
par(mfrow=c(3,2))
Frozen.Exp<-histDist(data$Frozen, family=EXP, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Exponential Distributed")
Frozen.Ga<-histDist(data$Frozen, family=GA, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Gamma Distributed")
Frozen.Gg<-histDist(data$Frozen, family=GG, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Generalized Gamma Distributed")
Frozen.Ig<-histDist(data$Frozen, family=IG, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Invers Gaussian Distributed")
Frozen.Logn<-histDist(data$Frozen, family=LOGNO, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Log-Normal Distributed")
Frozen.Wei<-histDist(data$Frozen, family=WEI, xlab="Annual Spending on Frozen Product", nbins=50, main="Spending on Frozen Product_Weibull Distributed")

data.frame(row.names=c("Exponential","Gamma","Generalized Gamma","Inverse Gaussian","Log-normal","Weibull"),
           df=c(Frozen.Exp$df.fit,Frozen.Ga$df.fit,Frozen.Gg$df.fit,Frozen.Ig$df.fit,Frozen.Logn$df.fit,Frozen.Wei$df.fit),
           AIC=c(AIC(Frozen.Exp),AIC(Frozen.Ga),AIC(Frozen.Gg),AIC(Frozen.Ig),AIC(Frozen.Logn),AIC(Frozen.Wei)),
           SBC=c(Frozen.Exp$sbc,Frozen.Ga$sbc,Frozen.Gg$sbc,Frozen.Ig$sbc,Frozen.Logn$sbc,Frozen.Wei$sbc),
           LogLik=c(logLik(Frozen.Exp),logLik(Frozen.Ga),logLik(Frozen.Gg),logLik(Frozen.Ig),logLik(Frozen.Logn),logLik(Frozen.Wei)))
#Accordind to the 3 criterions the best model is "Generalized Gamma".

#But it's possible compare other model in order to find the model which best fit our data using the function:
fitDist(data$Frozen,type="realplus")
#So, according to "fitDist" function, the best model is "Box-Cox-Cole-Green"

par(mfrow= c(1,1))
Frozen.BCCG<-histDist(data$Frozen, family=BCCG, xlab="Annual Spending on Frozen Product", nbins=50, main="Annual Spending on Frozen Product_Box-Cox-Cole-Green Distributed")
data.frame(row.names="Box-Cox-Cole-Green", df=Frozen.BCCG$df.fit, AIC=AIC(Frozen.BCCG),
           SBC=Frozen.BCCG$sbc, LogLik=logLik(Frozen.BCCG))
#According to the criterions and the "fitDist" function, the potentially best model is "Box-Cox-Cole-Green".

#MIXTURE MODEL for Frozen with k=2
#It's possible to combine two or more distribution and compute a mixture model that can fit better the
#distribution. In this case two gaussian distribution are used, so K=2.
set.seed(123)
par(mfrow= c(1,5))
F.mix.GA<-gamlssMXfits(n=5,data$Frozen~1,family=GA,K=2,data=NULL)

#estimate mu and sigma in the group 1 and 2
mu1<-exp(F.mix.GA[["models"]][[1]][["mu.coefficients"]])
mu2<-exp(F.mix.GA[["models"]][[2]][["mu.coefficients"]])
sigma1<-exp(F.mix.GA[["models"]][[1]][["sigma.coefficients"]])
sigma2<-exp(F.mix.GA[["models"]][[2]][["sigma.coefficients"]])

par(mfrow= c(1,1))
hist(data$Frozen,breaks=50,freq=FALSE,xlab="Frozen",main="Frozen - mixture of two Gamma models")
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F.mix.GA[["prob"]][1]*dGA(seq(min(data$Frozen), max(data$Frozen),length=length(data$Frozen)), mu=mu1, sigma=sigma1),lty=2,lwd=3,col=2)
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu2, sigma=sigma2), lty=2,lwd=3,col=3)
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F.mix.GA[["prob"]][1]*dGA(seq(min(data$Frozen), max(data$Frozen),length=length(data$Frozen)), mu=mu1, sigma=sigma1)+
        F.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu2, sigma= sigma2),
      lty=1,lwd=3,col=1)

#MIXTURE MODEL for Frozen with k=3
set.seed(123)
par(mfrow= c(1,5))
F3.mix.GA<-gamlssMXfits(n=5,data$Frozen~1,family=GA,K=3,data=NULL)

#estimate mu and sigma in the group 1 and 2
mu1<-exp(F3.mix.GA[["models"]][[1]][["mu.coefficients"]])
mu2<-exp(F3.mix.GA[["models"]][[2]][["mu.coefficients"]])
mu3<-exp(F3.mix.GA[["models"]][[3]][["mu.coefficients"]])
sigma1<-exp(F3.mix.GA[["models"]][[1]][["sigma.coefficients"]])
sigma2<-exp(F3.mix.GA[["models"]][[2]][["sigma.coefficients"]])
sigma3<-exp(F3.mix.GA[["models"]][[3]][["sigma.coefficients"]])

par(mfrow= c(1,1))
hist(data$Frozen,breaks=50,freq=FALSE,xlab="Frozen",main="Frozen - mixture of three Gamma models")
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F3.mix.GA[["prob"]][1]*dGA(seq(min(data$Frozen), max(data$Frozen),length=length(data$Frozen)), mu=mu1, sigma=sigma1),lty=2,lwd=3,col=2)
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F3.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu2, sigma=sigma2), lty=2,lwd=3,col=3)
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F3.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu3, sigma=sigma3), lty=2,lwd=3,col=4)
lines(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen))
      ,F3.mix.GA[["prob"]][1]*dGA(seq(min(data$Frozen), max(data$Frozen),length=length(data$Frozen)), mu=mu1, sigma=sigma1)+
        F3.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu2, sigma=sigma2)+
        F3.mix.GA[["prob"]][2]*dGA(seq(min(data$Frozen),max(data$Frozen),length=length(data$Frozen)), mu=mu3, sigma=sigma3),
      lty=1,lwd=3,col=1)

data.frame(row.names=c("Gamma mixtures with K=2","Gamma mixtures with K=3"), AIC=c(AIC(F.mix.GA),AIC(F3.mix.GA)),
           SBC=c(F.mix.GA$sbc,F3.mix.GA$sbc), LogLik=c(logLik(F.mix.GA),logLik(F3.mix.GA)))
LR.test(F.mix.GA,F3.mix.GA)
#Using the LR.test in order to check if the mixture model of three gamma distributions brings enough
#information to justify the higher number of parameters. Due to the p-value<0.05 the model with K=3 is
#effectively the best one for the LogLik criterion.

LR.test(Frozen.BCCG,F3.mix.GA)
#In this case the single "Box-Cox-Cole-Green" is the best model according the LR test


##About Annual Spending on Frozen Product
#is a numerical continuous variable, defined on the interval [0,+∞)
summary(data$Detergents_Paper)
kurtosis(data$Detergents_Paper)
skewness(data$Detergents_Paper)
jarque.test(data$Detergents_Paper)

par(mfrow= c(1,1))
box<-boxplot(data$Detergents_Paper)
box$out
data[which(data$Detergents_Paper%in% box$out),]
#In this case all the outliers come from the Retail channel, 
#this might be releated that Retail is characterized by 
#high values of annual spending score on Detergents and Paper Product than the other Channel.

table(data$Detergents_Paper)
length(unique(data$Detergents_Paper))
quantile(data$Detergents_Paper, 0.25)
quantile(data$Detergents_Paper, 0.50)
quantile(data$Detergents_Paper, 0.75)

par(mfrow= c(1,1))
hist(data$Detergents_Paper, main="Annual Spending on Detergent and Paper Product", ylim=c(0,3e-04), col="red", freq = FALSE)
lines(density(data$Detergents_Paper), col="black")


#Model Fitting of Annual Spending on Detergent and Paper
par(mfrow=c(3,2))
DP.Exp<-histDist(data$Detergents_Paper, family=EXP, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Exponential Distributed")
DP.Ga<-histDist(data$Detergents_Paper, family=GA, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Gamma Distributed")
DP.Gg<-histDist(data$Detergents_Paper, family=GG, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Generalized Gamma Distributed")
DP.Ig<-histDist(data$Detergents_Paper, family=IG, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Invers Gaussian Distributed")
DP.Logn<-histDist(data$Detergents_Paper, family=LOGNO, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Log-Normal Distributed")
DP.Wei<-histDist(data$Detergents_Paper, family=WEI, xlab="Annual Spending Detergents_Paper Product", nbins=50, main="Spending on Detergents_Paper Product_Weibull Distributed")

data.frame(row.names=c("Exponential","Gamma","Generalized Gamma","Inverse Gaussian","Log-normal","Weibull"),
           df=c(DP.Exp$df.fit,DP.Ga$df.fit,DP.Gg$df.fit,DP.Ig$df.fit,DP.Logn$df.fit,DP.Wei$df.fit),
           AIC=c(AIC(DP.Exp),AIC(DP.Ga),AIC(DP.Gg),AIC(DP.Ig),AIC(DP.Logn),AIC(DP.Wei)),
           SBC=c(DP.Exp$sbc,DP.Ga$sbc,DP.Gg$sbc,DP.Ig$sbc,DP.Logn$sbc,DP.Wei$sbc),
           LogLik=c(logLik(DP.Exp),logLik(DP.Ga),logLik(DP.Gg),logLik(DP.Ig),logLik(DP.Logn),logLik(DP.Wei)))

#According to the criterions the best model is the "Generalized Gamma".

#But it's possible compare other model in order to find the model which best fit our data using the function:
fitDist(data$Detergents_Paper, type="realplus")

par(mfrow= c(1,1))
DP.BCPEo<-histDist(data$Detergents_Paper, family=BCPEo, xlab="Annual Spending on Detergents_Paper Product", nbins=50, main="Detergents and Paper Product_Box-Cox Power Exponential-orig. Distributed")
data.frame(row.names="Box-Cox Power Exponential-orig.", df=DP.BCPEo$df.fit, AIC=AIC(DP.BCPEo),
           SBC=DP.BCPEo$sbc, LogLik=logLik(DP.BCPEo))
#So according to the all criterion and to the "fitDist" function, the potentially best model is the "Box-Cox Power Exponential-orig."


#MIXTURE MODEL for Detergents_Paper with k=2
#It's possible to combine two or more distribution and compute a mixture model that can fit better the
#distribution. In this case two gaussian distribution are used, so K=2.
set.seed(123)
par(mfrow= c(1,5))
DP.mix.GA<-gamlssMXfits(n=5,data$Detergents_Paper~1,family=GA,K=2,data=NULL)

#estimate mu and sigma in the group 1 and 2
mu1<-exp(DP.mix.GA[["models"]][[1]][["mu.coefficients"]])
sigma1<-exp(DP.mix.GA[["models"]][[1]][["sigma.coefficients"]])
mu2<-exp(DP.mix.GA[["models"]][[2]][["mu.coefficients"]])
sigma2<-exp(DP.mix.GA[["models"]][[2]][["sigma.coefficients"]])

par(mfrow= c(1,1))
hist(data$Detergents_Paper,breaks=50,freq=FALSE,xlab="Detergents_Paper",main="Detergents_Paper - mixture of two gamma models")
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP.mix.GA[["prob"]][1]*dGA(seq(min(data$Detergents_Paper), max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu1, sigma=sigma1),lty=2,lwd=3,col=2)
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu2, sigma=sigma2), lty=2,lwd=3,col=3)
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP.mix.GA[["prob"]][1]*dGA(seq(min(data$Detergents_Paper), max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu1, sigma=sigma1)+
        DP.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu2, sigma=sigma2),
      lty=1,lwd=3,col=1)


#MIXTURE MODEL for Detergents_Paper with k=3
set.seed(123)
par(mfrow= c(1,5))
DP3.mix.GA<-gamlssMXfits(n=5,data$Detergents_Paper~1,family=GA,K=3,data=NULL)

#estimate mu and sigma in the group 1 and 2
mu1<-exp(DP3.mix.GA[["models"]][[1]][["mu.coefficients"]])
sigma1<-exp(DP3.mix.GA[["models"]][[1]][["sigma.coefficients"]])
mu2<-exp(DP3.mix.GA[["models"]][[2]][["mu.coefficients"]])
sigma2<-exp(DP3.mix.GA[["models"]][[2]][["sigma.coefficients"]])
mu3<-exp(DP3.mix.GA[["models"]][[3]][["mu.coefficients"]])
sigma3<-exp(DP3.mix.GA[["models"]][[3]][["sigma.coefficients"]])

par(mfrow= c(1,1))
hist(data$Detergents_Paper,breaks=50,freq=FALSE,xlab="Detergents_Paper",main="Detergents_Paper - mixture of three gamma models")
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP3.mix.GA[["prob"]][1]*dGA(seq(min(data$Detergents_Paper), max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu1, sigma=sigma1),lty=2,lwd=3,col=2)
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP3.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu2, sigma=sigma2), lty=2,lwd=3,col=3)
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP3.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu3, sigma=sigma3), lty=2,lwd=3,col=4)
lines(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper))
      ,DP3.mix.GA[["prob"]][1]*dGA(seq(min(data$Detergents_Paper), max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu1, sigma=sigma1)+
        DP3.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu2, sigma=sigma2)+
        DP3.mix.GA[["prob"]][2]*dGA(seq(min(data$Detergents_Paper),max(data$Detergents_Paper),length=length(data$Detergents_Paper)), mu=mu3, sigma=sigma3),
      lty=1,lwd=3,col=1)

data.frame(row.names=c("Gamma mixtures with K=2","Gamma mixtures with K=3"), AIC=c(AIC(DP.mix.GA),AIC(DP3.mix.GA)),
           SBC=c(DP.mix.GA$sbc,DP3.mix.GA$sbc), LogLik=c(logLik(DP.mix.GA),logLik(DP3.mix.GA)))
LR.test(DP.mix.GA,DP3.mix.GA)
#Using the LR.test in order to check if the mixture model of three gamma distributions brings enough
#information to justify the higher number of parameters. Due to the p-value<0.05 the model with K=3 is
#effectively the best method according to the AIC and LogLik criterions.


##Management of Outliers
data <- read_csv("C:/Users/acer/Downloads/Wholesale customers data.csv")
#Outliers from Fresh
par(mfrow= c(1,2))
boxF<-boxplot(data$Fresh)
#Outliers
boxF$out
#Located
#sapply(data$Fresh, function(x) x %in% boxF$out)
#Replace Outliers with the median
data$Fresh[data$Fresh %in% boxF$out]<- median(data$Fresh)
boxplot(data$Fresh)

#Outliers from Milk
par(mfrow= c(1,2))
boxM<-boxplot(data$Milk)
boxM$out
data$Milk[data$Milk %in% boxM$out]<- median(data$Milk)
boxplot(data$Milk)

#Outliers from Grocery
par(mfrow= c(1,2))
boxG<-boxplot(data$Grocery)
boxG$out
data$Grocery[data$Grocery %in% boxG$out]<- median(data$Grocery)
boxplot(data$Grocery)

#Outliers from Frozen
par(mfrow= c(1,2))
boxFro<-boxplot(data$Frozen)
boxFro$out
data$Frozen[data$Frozen %in% boxFro$out]<- median(data$Frozen)
boxplot(data$Frozen)

#Outliers from Detergents_Paper
par(mfrow= c(1,2))
boxD<-boxplot(data$Detergents_Paper)
boxD$out
data$Detergents_Paper[data$Detergents_Paper %in% boxD$out]<- median(data$Detergents_Paper)
boxplot(data$Detergents_Paper)

#Outliers from Delicassen
par(mfrow= c(1,2))
boxDl<-boxplot(data$Delicassen)
boxDl$out
data$Delicassen[data$Delicassen %in% boxDl$out]<- median(data$Delicassen)
boxplot(data$Delicassen)


#PCA ANALYSIS
#Before to proceed with the PCA analysis, 
#we have to select the numeric variable and then scale the data
df<-data[3:8]
summary(df)

#Each originals variable may have different mean, 
#so can be usefull centered at zero each variable for the PCA 
#in order to compare each principal componenet to the mean straightforward
scale_df<-scale(df)
summary(scale_df)

#we can compute the Correlation Matrix
cor(scale_df)

par(mfrow= c(1,1))
corrplot(cor(scale_df), type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)
#The corrplot above show us important stuff:
#•a high positive linear correlation between Detergents_Paper_ and Grocery around 0,71
#•a positive linear correlation between Milk and Grocery around 0,62
#•a positive linear correlation between Milk and Detergents_Paper_ around 0.54

#Calculating the eigenvalues and the eigenvector
cov(scale_df)
data.eigen<-eigen(cov(scale_df))
str(data.eigen)

##Selecting the PCs
#There are 3 different method to select the right number o PCs

#Proportion of Variance Explained
PVE<- data.eigen$values/sum(data.eigen$values)
round(PVE, 3)
#The first principal component explain the 39,7% of the variability
#The second principal component explain the 21,5% of the variability
#The third principal component explain the 14,5% of the variability
#Together the first 3 PCs explains the 75,7% of the variability

#It' s convenient to plot the PVE and the CPVE (Cumulative Proportion of Variance Explained) in a scree plot.
PVEplot<-qplot(c(1:6), PVE) + geom_line() + xlab("Principal Component") + ylab("PVE") +ggtitle("Scree Plot") +ylim(0, 1)
cumPVE <- qplot(c(1:6), cumsum(PVE)) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("CMPVE") + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)
grid.arrange(PVEplot, cumPVE, ncol = 2)
#We can estimate the number of PCs by looking for the "elbow point" in the scree plot, 
#where the PVE significantly drops off. In this case the perfect number o PCs is equal 3.

#Kaiser's Rule
#This rule suggest to take as many PCs as are the eigenvalues larger than 1.
data.eigen$value[data.eigen$value>1]
#According to this rule can be retained the first two PCs

#Principal Component Loadings and Biplot
phi<-data.eigen$vector[,1:3]
phi
#The eigenvector calculated are unique up to a sign flip. 
#So is important choice the sign of each eigenvector in order to make the interpretation easier. 
#By default eigentvector are computed in the negative direction.

row.names(phi)<- c("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")
colnames(phi)<- c("PC1", "PC2","PC3")
phi1<-(phi*-1)
phi1
#The first loading vector are influenced in equal way from Milk,Grocery and Detergents_Paper, 
#while much less by Fresh,Frozen and Delicassen.

#The second loading vector place most of the weight from Fronzen, and also from Fresh and Delicassen, 
#with some oppisite influence from Detergents_paper and Grocery.

#The third loading vector take most of the weight from Fresh but in a strong negative way, 
#while is positive influenced from Delicassen,Milk,Frozen.

PC1<- scale_df%*%phi1[,1]
PC2<- scale_df%*%phi1[,2]
PC3<- scale_df%*%phi1[,3]
PC<- data.frame(Type_Product=row.names(data), PC1,PC2,PC3)
head(PC)
#We have computed the first, the second, and the third principal components for each observation.

pr.out=prcomp(df, scale=TRUE)
pr.out$rotation<- -pr.out$rotation
pr.out$rotation[,1:3]
pr.out$x<- -pr.out$x


par(mfrow= c(1,1))
fviz_pca_ind(pr.out,col.ind="cos2" )

par(mfrow= c(1,1))
fviz_pca_ind(pr.out, label="none", habillage=data$Channel)
pca3d(pr.out, group=data$Region)
#we will use channel for the group
pca3d(pr.out, group=data$Channel, show.ellipses=TRUE,
      ellipse.ci=0.75, show.plane=FALSE)
pca2d(pr.out, group=data$Channel, biplot=TRUE, biplot.vars=3)
biplot(pr.out, scale=0, col=c("lightblue","orange"))



## CLUSTER ANALYSIS

#Before applying any clustering method it's important to evaluate whether the data set contains meaningful
#clusters or not.

#Assessing clustering method
#Before analyzing these two method in details, it's necessary to create a a uniform random data set from the
#original data. If the dataset contains clusters it will be possible to see the difference from a dataset created
#through a uniform distribution of data.

random_df<-apply(df,2,function(x){runif(length(x), min(x), max(x))})
random_df<-as.data.frame(random_df)
srandom_df<-scale(random_df)


pairs(scale_df, gap=0, pch=16)
pairs(scale_df, main="Channel Classification", pch = 21, bg = c("red", "blue")[unclass(data$Channel)])
pairs(scale_df, main="Region Classification", pch = 21, bg = c("red", "blue","green")[unclass(data$Region)])

pairs(srandom_df, pch=21 ,bg = c("red", "blue")[unclass(data$Channel)])

#It can be seen that the standarized randomly generated uniform data do not contain meaningful clusters.

pca_graph<-fviz_pca_ind(pr.out,title="PCA - dataset", habillage=data$Channel, geom = "point", ggtheme=theme_classic(), legend="bottom")
random_pca<-fviz_pca_ind(prcomp(srandom_df),title="PCA - random dataset", habillage=data$Channel, geom = "point", ggtheme=theme_classic(), legend="bottom")
grid.arrange(pca_graph, random_pca, ncol=2)

pca_3d<-pca3d(pr.out, group=data$Channel)
pca_r3d<-pca3d(prcomp(srandom_df), group=data$Channel)


## Visual method
D_euc<-dist(scale_df, method="euclidean")
D_man<-dist(scale_df, method="manhattan")
D_min<-dist(scale_df, method="minkowski")
D_mixed<-daisy(scale(data))

VAT.euc<-fviz_dist(D_euc, show_labels=FALSE)+ labs(title="Euclidean distance")
VAT.man<-fviz_dist(D_man, show_labels=FALSE)+ labs(title="Manhattan distance")
VAT.min<-fviz_dist(D_min, show_labels=FALSE)+ labs(title="Minkowski distance")
VAT.mix<-fviz_dist(D_mixed, show_labels=FALSE)+ labs(title="Mixed distance")

VAT.ran<-fviz_dist(dist(srandom_df), show_labels=FALSE)+ labs(title="Random Data")
grid.arrange(VAT.euc,VAT.man,VAT.min,VAT.mix,VAT.ran, ncol=2)

#the color level is proportional to the value of the dissimilarity btw observations:
#red means high similarity, while blue means low similarity
#in this case the dissimilarity matrix image confirms that there are 
#clusters structure in our data but not in the random one

# STATISTICAL METHOD

set.seed(123)
hopkins::hopkins(scale_df, m=nrow(scale_df)-1)
hopkins::hopkins(srandom_df, m=nrow(srandom_df)-1)
#By looking at the Hopkins statistic it's possible to see that the dataset has an high value (close to 1), which
#means that there are clusters. Instead for the random data set the result is a lower value.
#It's also possible to use the "get_clust_tendency" function that provide togheter the Hopkins statistic and the
#VAT algorithm result. Below the function is applied to the scaled data set and to the scaled random data set,
#in order to compare the results:

get_clust_tendency(scale_df, n=nrow(scale_df)-1, graph=TRUE, seed=123)
get_clust_tendency(srandom_df, n=nrow(srandom_df)-1, graph=TRUE, seed=123)


## CLUSTERING ANALYSIS
#average linkage-euclidean dist
nb1<-NbClust(scale_df, distance="euclidean", min.nc=2, max.nc=15, method="average")
summary(nb1)
t <- table(nb1$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="euclidean", hc_method="average")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="euclidean", hc_method="average")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="euclidean", hc_method="average")+labs(subtitle = "Gap Statistic Method")

#according with this method, 3 clusters is the most suggested choice
avg.euc.hc<-hclust(d=D_euc, method="average")
par(mfrow=c(1,1))
fviz_dend(avg.euc.hc, k=3, cex=0.5,k_colors=c("orange","lightblue","red"),rect=TRUE)

avg.euc.coph<-cophenetic(avg.euc.hc)
cor(D_euc, avg.euc.coph)

grp.avg.euc<-cutree(avg.euc.hc, k=3)
table(grp.avg.euc)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.avg.euc, col=c("orange","black","green")[grp.avg.euc])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.avg.euc), palette=c("orange","black","green"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())
###internal validation silhouette and dunn index

avg.res<-eclust(scale_df, "hclust", k=3, hc_metric="euclidean", hc_method="average", graph=FALSE)
fviz_silhouette(avg.res, palette="jco", ggtheme=theme_classic())

silinfo.avg<-avg.res$silinfo
neg_sil.avg<-which(silinfo.avg$widths[,"sil_width"]<0)

silinfo.avg$avg.width
silinfo.avg$clus.avg.widths
silinfo.avg$widths[neg_sil.avg,,drop=FALSE]

avg.link<-cluster.stats(D_euc, avg.res$cluster)
avg.link$dunn

#External validation
#confusion matrix
table(data$Channel, avg.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.avg<-cluster.stats(d=D_euc, tipes, avg.res$cluster)
clust_stats.avg$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.avg$vi

#average linkage method (Manhattan dist)
nb2<-NbClust(scale_df, distance="manhattan", method="average")
summary(nb2)
t2 <- table(nb2$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t2,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="manhattan", hc_method="average")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="manhattan", hc_method="average")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="manhattan", hc_method="average")+labs(subtitle = "Gap Statistic Method")
#according with this method, three clusters is the most suggested choice
avg.man.hc<-hclust(d=D_man, method="average")

par(mfrow=c(1,1))
fviz_dend(avg.man.hc, k=3, cex=0.5,k_colors=c("orange","lightblue","green"),rect=TRUE)

avg.man.coph<-cophenetic(avg.man.hc)
cor(D_euc, avg.man.coph)

grp.man<-cutree(avg.man.hc, k=3)
table(grp.man)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.man, col=c("orange","black","yellow")[grp.man])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.man), palette=c("orange","red","green"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())
#Internal Validation
man.res<-eclust(scale_df, "hclust", k=3, hc_metric="manhattan", hc_method="average", graph=FALSE)
fviz_silhouette(man.res, palette="jco", ggtheme=theme_classic())

silinfo.man<-man.res$silinfo
neg_sil.man<-which(silinfo.man$widths[,"sil_width"]<0)

silinfo.man$avg.width
silinfo.man$clus.avg.widths
silinfo.man$widths[neg_sil.man,,drop=FALSE]

man.link<-cluster.stats(D_man, man.res$cluster)
man.link$dunn

#External validation
#confusion matrix
table(data$Channel, man.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.man<-cluster.stats(d=D_man, tipes, man.res$cluster)
clust_stats.man$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.man$vi

#complete linkage method (Euclidean dist)-more robust
nb3<-NbClust(scale_df, distance="euclidean", method="complete")
summary(nb3)
t3 <- table(nb3$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t3,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="euclidean", hc_method="complete")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="euclidean", hc_method="complete")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="euclidean", hc_method="complete")+labs(subtitle = "Gap Statistic Method")
#according with this method, 2 clusters is the most suggested choice
com.euc.hc<-hclust(d=D_euc, method="complete")

par(mfrow=c(1,1))
fviz_dend(com.euc.hc, k=2, cex=0.5,k_colors=c("orange","lightblue"),rect=TRUE)

com.euc.coph<-cophenetic(com.euc.hc)
cor(D_euc, com.euc.coph)

grp.c.euc<-cutree(com.euc.hc, k=2)
table(grp.c.euc)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.c.euc, col=c("orange","red")[grp.c.euc])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.c.euc), palette=c("red","green"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())
#Internal Validation
c.euc.res<-eclust(scale_df, "hclust", k=2, hc_metric="euclidean", hc_method="complete", graph=FALSE)
fviz_silhouette(c.euc.res, palette="jco", ggtheme=theme_classic())

silinfo.c.euc<-c.euc.res$silinfo
neg_sil.c.euc<-which(silinfo.c.euc$widths[,"sil_width"]<0)

silinfo.c.euc$avg.width
silinfo.c.euc$clus.avg.widths
silinfo.c.euc$widths[neg_sil.c.euc,,drop=FALSE]

c.euc.link<-cluster.stats(D_euc, c.euc.res$cluster)
c.euc.link$dunn

#External validation
#confusion matrix
table(data$Channel, c.euc.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.c.euc<-cluster.stats(d=D_euc, tipes, c.euc.res$cluster)
clust_stats.c.euc$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.c.euc$vi

#complete linkage method (Manhattan dist)-more robust
nb4<-NbClust(scale_df, distance="manhattan", method="complete")
summary(nb4)
t4 <- table(nb4$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t4,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="manhattan", hc_method="complete")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="manhatta", hc_method="complete")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="manhattan", hc_method="complete")+labs(subtitle = "Gap Statistic Method")
#according with this method, 2 clusters is the most suggested choice
com.man.hc<-hclust(d=D_man, method="complete")

par(mfrow=c(1,1))
fviz_dend(com.man.hc, k=2, cex=0.5,k_colors=c("orange","lightblue"),rect=TRUE)

com.man.coph<-cophenetic(com.man.hc)
cor(D_man, com.man.coph)

grp.c.man1<-cutree(com.man.hc, k=2)
table(grp.c.man1)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.c.man1, col=c("orange","red")[grp.c.man1])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.c.man1), palette=c("red","green"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())

#Internal Validation
c.man.res<-eclust(scale_df, "hclust", k=2, hc_metric="manhattan", hc_method="complete", graph=FALSE)
fviz_silhouette(c.man.res, palette="jco", ggtheme=theme_classic())

silinfo.c.man<-c.man.res$silinfo
neg_sil.c.man<-which(silinfo.c.man$widths[,"sil_width"]<0)

silinfo.c.man$avg.width
silinfo.c.man$clus.avg.widths
silinfo.c.man$widths[neg_sil.c.man,,drop=FALSE]

c.man.link<-cluster.stats(D_man, c.man.res$cluster)
c.man.link$dunn

#External validation
#confusion matrix
table(data$Channel, c.man.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.c.man<-cluster.stats(d=D_man, tipes, c.man.res$cluster)
clust_stats.c.man$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.c.man$vi

#complete linkage method (Minkowski dist)-more robust
nb5<-NbClust(scale_df, distance="minkowski", method="complete")
summary(nb5)
t5 <- table(nb5$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t5,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="minkowski", hc_method="complete")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="minkowski", hc_method="complete")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="minkowski", hc_method="complete")+labs(subtitle = "Gap Statistic Method")
#according with this method, 2 clusters is the most suggested choice
com.min.hc<-hclust(d=D_min, method="complete")

par(mfrow=c(1,1))
fviz_dend(com.min.hc, k=2, cex=0.5,k_colors=c("orange","lightblue"),rect=TRUE)

com.min.coph<-cophenetic(com.min.hc)
cor(D_min, com.min.coph)

grp.c.min<-cutree(com.min.hc, k=2)
table(grp.c.min)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.c.min, col=c("orange","red")[grp.c.min])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.c.min), palette=c("red","green"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())

#Internal Validation
c.min.res<-eclust(scale_df, "hclust", k=2, hc_metric="minkowski", hc_method="complete", graph=FALSE)
fviz_silhouette(c.min.res, palette="jco", ggtheme=theme_classic())

silinfo.c.min<-c.min.res$silinfo
neg_sil.c.min<-which(silinfo.c.min$widths[,"sil_width"]<0)

silinfo.c.min$avg.width
silinfo.c.min$clus.avg.widths
silinfo.c.min$widths[neg_sil.c.min,,drop=FALSE]

c.min.link<-cluster.stats(D_min, c.min.res$cluster)
c.min.link$dunn

#External validation
#confusion matrix
table(data$Channel, c.min.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.c.min<-cluster.stats(d=D_min, tipes, c.min.res$cluster)
clust_stats.c.min$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.c.min$vi

#Ward's method (Manhattan dist)
nb6<-NbClust(scale_df, distance="manhattan", method="ward.D2")
summary(nb6)
t6 <- table(nb6$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t6,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="wss", hc_metric="manhattan", hc_method="ward.D2")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="silhouette", hc_metric="manhattan", hc_method="ward.D2")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, hcut, method="gap_stat", nboot=500, hc_metric="manhattan", hc_method="ward.D2")+labs(subtitle = "Gap Statistic Method")
#according with this method, 4 clusters is the most suggested choice
ward.man.hc<-hclust(d=D_man, method="ward.D2")

par(mfrow=c(1,1))
fviz_dend(ward.man.hc, k=4, cex=0.5,k_colors=c("orange","lightblue","green","red"),rect=TRUE)

ward.man.coph<-cophenetic(ward.man.hc)
cor(D_man, ward.man.coph)

grp.w<-cutree(ward.man.hc, k=4)
table(grp.w)

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=grp.w, col=c("orange","lightblue","green","yellow")[grp.w])

par(mfrow=c(1,1))
fviz_cluster(list(data=scale_df, cluster=grp.w), palette=c("orange","lightblue","green","yellow"),
             ellipse.type="convex",repel=TRUE, show.clust.cent=FALSE, ggtheme= theme_minimal())
#Internal Validation
w.res<-eclust(scale_df, "hclust", k=4, hc_metric="manhattan", hc_method="ward.D2", graph=FALSE)
fviz_silhouette(w.res, palette="jco", ggtheme=theme_classic())

silinfo.w<-w.res$silinfo
neg_sil.w<-which(silinfo.w$widths[,"sil_width"]<0)

silinfo.w$avg.width
silinfo.w$clus.avg.widths
silinfo.w$widths[neg_sil.w,,drop=FALSE]

w.link<-cluster.stats(D_man, w.res$cluster)
w.link$dunn

#External validation
#confusion matrix
table(data$Channel, w.res$cluster)
#rand index
tipes<-as.numeric(data$Channel)
clust_stats.w<-cluster.stats(d=D_euc, tipes, w.res$cluster)
clust_stats.w$corrected.rand
#The ARI provides an index that is close to 0 because it takes into account the chance of overlap. 
#In addition, note that the ARI is a negative value indicating that the amount of overlap is less than expected.
##Meila's VI Index
clust_stats.w$vi

#K-means method (Euclidean dist)
nb7<-NbClust(scale_df, distance="euclidean", method="kmeans")
summary(nb7)
t7 <- table(nb7$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t7,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="wss")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="silhouette")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="gap_stat", nboot=500)+labs(subtitle = "Gap Statistic Method")
#according with this method, 2 clusters is the most suggested choice

set.seed(123)
km.euc<-kmeans(scale_df, 2, iter.max=100, nstart=50)
print(km.euc)
#Visualize results
km.cl<-km.euc$cluster

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=km.cl, col=c("orange","lightblue")[km.cl])
#Profiling Cluster
aggregate(data[3:8], by=list(cluster=km.euc$cluster), mean)
#explain 
#In cluster 1 it seems that units have higher value of all the variables except for Annual Spending
# for Fresh and Frozen Product, this variables doesn't seem good to clusterize data.

#Adding the classification to the original dataset:
data.km<-cbind(data, cluster=km.euc$cluster)
head(data.km)

par(mfrow=c(1,1))
fviz_cluster(km.euc, data=scale_df, palette=c("orange","lightblue"),ellipse.type="euclid", star.plot=TRUE,
             repel = TRUE, ggtheme=theme_minimal())

#INternal Validation
km.euc.ec<-eclust(scale_df, "kmeans", k=2, hc_metric="euclidean", graph=FALSE)
fviz_silhouette(km.euc.ec, palette="jco", ggtheme=theme_classic())

silinfo.km.euc<-km.euc.ec$silinfo
silinfo.km.euc$avg.width
silinfo.km.euc$clus.avg.widths
neg_sil.km.euc<-which(silinfo.km.euc$widths[,"sil_width"]<0)
silinfo.km.euc$widths[neg_sil.km.euc,,drop=FALSE]

k.link<-cluster.stats(D_euc, km.euc.ec$cluster)
k.link$dunn

#External validation for k=2

table(data$Channel, km.euc.ec$cluster)
#In this case there' s no perfect agreement between the external information and the cluster structure

tipes<-as.numeric(data$Channel)
clust_stats.km.euc<-cluster.stats(d=D_euc, tipes, km.euc$cluster)
clust_stats.km.euc$corrected.rand

#According to the correct Rand index there is a good agreement between the seed types and the cluster
#solution

clust_stats.km.euc$vi

#K-means method (Manhattan dist)
nb8<-NbClust(scale_df, distance="manhattan", method="kmeans")
summary(nb8)
t8 <- table(nb8$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t8,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="wss", diss= dist(scale_df, method="manhattan"))+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="silhouette", diss= dist(scale_df, method="manhattan"))+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, kmeans, method="gap_stat",diss= dist(scale_df, method="manhattan"), nboot=500)+labs(subtitle = "Gap Statistic Method")

#according with this method, 2 clusters is the most suggested choice

set.seed(123)
km.man<-kmeans(scale_df, 2, iter.max = 100, nstart=50)
print(km.man)
#Visualize results
km.man.cl<-km.man$cluster

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=km.man.cl, col=c("orange","lightblue")[km.man.cl])
#Profiling Cluster
aggregate(data[3:8], by=list(cluster=km.man$cluster), mean)
#explain 
#In cluster 1 it seems that units have higher value of all the variables except for Annual Spending
# for Fresh and Frozen Product, this variables doesn't seem good to clusterize data.

#Adding the classification to the original dataset:
data.km.man<-cbind(data, cluster=km.man$cluster)
head(data.km.man)

par(mfrow=c(1,1))
fviz_cluster(km.man, data=scale_df, palette=c("orange","lightblue"),ellipse.type="euclid", star.plot=TRUE,
             repel = TRUE, ggtheme=theme_minimal())

#INternal Validation
km.man.ec<-eclust(scale_df, "kmeans", k=2, hc_metric="manhattan", graph=FALSE)
fviz_silhouette(km.man.ec, palette="jco", ggtheme=theme_classic())

silinfo.km.man<-km.man.ec$silinfo
silinfo.km.man$avg.width
silinfo.km.man$clus.avg.widths
neg_sil.km.man<-which(silinfo.km.man$widths[,"sil_width"]<0)
silinfo.km.man$widths[neg_sil.km.man,,drop=FALSE]

k.link2<-cluster.stats(D_man, km.man.ec$cluster)
k.link2$dunn

#External validation for k=2

table(data$Channel, km.man.ec$cluster)
#In this case there' s no perfect agreement between the external information and the cluster structure

tipes<-as.numeric(data$Channel)
clust_stats.km.man<-cluster.stats(d=D_man, tipes, km.man$cluster)
clust_stats.km.man$corrected.rand

#According to the correct Rand index there is a good agreement between the seed types and the cluster
#solution
clust_stats.km.man$vi

#PAM Method (Euclidean distance)
nb9<-NbClust(scale_df, distance="euclidean", method="centroid")
summary(nb9)
t9 <- table(nb9$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t9,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam, diss=D_euc, metric="euclidean", method="wss")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam,diss=D_euc, metric="euclidean", method="silhouette")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam, diss=D_euc, metric="euclidean", method="gap_stat", nboot=500)+labs(subtitle = "Gap Statistic Method")

#according with this method, 2 clusters is the most suggested choice

set.seed(123)
pam.euc<-pam(scale_df,2, metric="euclidean")
print(pam.euc)
pam.euc$clusinfo
pam.euc.cl<-pam.euc$clustering

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=pam.euc.cl, col=c("orange","lightblue")[pam.euc.cl])

par(mfrow=c(1,1))
fviz_cluster(pam.euc, data=scale_df, palette=c("orange","lightblue"),ellipse.type="t", star.plot=TRUE,
             repel = TRUE, ggtheme=theme_minimal())
#INTERNAL VALIDATION
pam.euc.ec<-eclust(scale_df, "pam", k=2, k.max=14, graph=FALSE, hc_metric="euclidean")
fviz_silhouette(pam.euc, palette="jco", ggtheme=theme_classic())
silinfo.pam.euc<-pam.euc$silinfo
silinfo.pam.euc$avg.width
silinfo.pam.euc$clus.avg.widths
neg_sil.pam.euc<-which(silinfo.pam.euc$widths[,"sil_width"]<0)
silinfo.pam.euc$widths[neg_sil.pam.euc,,drop=FALSE]

pam.euc.link<-cluster.stats(D_euc, pam.euc.ec$cluster)
pam.euc.link$dunn

#EXTERNAL VALIDATION: confusion matrix, correct Rand index, Meila's VI index
table(data$Channel, pam.euc.ec$cluster)
#In this case there' s no perfect agreement between the external information and the cluster structure

tipes<-as.numeric(data$Channel)
clust_stats.pam.euc<-cluster.stats(d=D_euc, tipes, pam.euc$cluster)
clust_stats.pam.euc$corrected.rand

#According to the correct Rand index there is a good agreement between the seed types and the cluster
#solution
clust_stats.pam.euc$vi

#PAM Method (Manhattan distance)
nb10<-NbClust(scale_df, distance="manhattan", method="centroid")
summary(nb10)
t10 <- table(nb10$Best.nc[1,])

par(mfrow= c(1,1))
barplot(t10,xlab="Number of Clusters",ylab="Number of Criteria",
        main="Number of Clusters chosen by 26 Criteria")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam, diss=D_man, metric="manhattan", method="wss")+geom_vline(xintercept = 4, linetype=2)+labs(subtitle = "Elbow Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam,diss=D_man, metric="manhattan", method="silhouette")+labs(subtitle = "Silhouette Method")

par(mfrow= c(1,1))
fviz_nbclust(scale_df, cluster::pam, diss=D_man, metric="manhattan", method="gap_stat", nboot=500)+labs(subtitle = "Gap Statistic Method")

#according with this method, 2 clusters is the most suggested choice

set.seed(123)
pam.man<-pam(scale_df,2, metric="manhattan")
print(pam.man)
pam.man$clusinfo
pam.man.cl<-pam.man$clustering

par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=pam.man.cl, col=c("red","blue")[pam.man.cl])

par(mfrow=c(1,1))
fviz_cluster(pam.man, data=scale_df, palette="jco",ellipse.type="t", star.plot=TRUE,
             repel = TRUE, ggtheme=theme_minimal())

#INTERNAL VALIDATION
pam.man.ec<-eclust(scale_df, "pam", k=2, k.max=14, graph=FALSE, hc_metric="manhattan")
fviz_silhouette(pam.man, palette="jco", ggtheme=theme_classic())
silinfo.pam.man<-pam.man$silinfo
silinfo.pam.man$avg.width
silinfo.pam.man$clus.avg.widths
neg_sil.pam.man<-which(silinfo.pam.man$widths[,"sil_width"]<0)
silinfo.pam.man$widths[neg_sil.pam.man,,drop=FALSE]

pam.man.link<-cluster.stats(D_man, pam.man.ec$cluster)
pam.man.link$dunn

#EXTERNAL VALIDATION: confusion matrix, correct Rand index, Meila's VI index
table(data$Channel, pam.man.ec$cluster)
#In this case there' s no perfect agreement between the external information and the cluster structure

tipes<-as.numeric(data$Channel)
clust_stats.pam.man<-cluster.stats(d=D_man, tipes, pam.man$cluster)
clust_stats.pam.man$corrected.rand

#According to the correct Rand index there is a good agreement between the seed types and the cluster
#solution
clust_stats.pam.man$vi


## Model Based Clustering
#Find the best parsimonious configuration of the Gaussian mixture distribution and the most appropriate
#number of K components.
model<-Mclust(scale_df, G=1:9, modelNames=NULL)
summary(model)

#soft assignment
head(round(model$z, 6), 30)
#hard assignment
head(model$classification, 30)

#best BIC Value
summary(model$BIC)

par(mfrow=c(1,1))
plot(model, what="BIC", ylim=range(model$BIC, na.rm=TRUE), legendArgs=list(x="bottomleft"))

par(mfrow=c(1,1))
fviz_mclust(model, "BIC", palette="jco")

#According with the BIC criterion the best choice is a model with seven components (clusters) and the model is
#VVI: variable volume, variable shape and orentation.
#Plotting the classification:
par(mfrow=c(1,1))
pairs(scale_df, gap=0, pch=16, col=model$classification)

par(mfrow=c(1,1))
fviz_mclust(model, "classification", geom="point", pointsize=1.5, palette="jco", main="Mod
Gaussian mixture model, K=7")

par(mfrow=c(1,1))
fviz_mclust(model, "uncertainty", palette="jco")
#Larger points indicate the more uncertain observations.

#External validation
#Confusion matrix
table(data$Channel, model$classification)
#In this case there' s no perfect agreement between the external information and the cluster structure
#because there are four clusters and three category. Anyway we can see that:

#Corrected Rand index
adjustedRandIndex(data$Channel, model$classification)
#According to the adjusted Rand index, the agreement between seeds type and the clusters obtained via
#model-based clustering, is poor. This is due to the fact that the number of clusters do not match the
#external labels.

#Best clustering method
#In order to chose the best model of clustering analysis among all the "clValid" is going to be used.
clmethod<-c("hierarchical","kmeans","pam")
rownames(scale_df)<-rownames(data)

#AVG 
avg.euc_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
metric="euclidean", method="average")
summary(avg.euc_valid)
#According to three of the seven indices the best method is the K-means with two clusters.

avg.man_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
                       metric="manhattan", method="average")
summary(avg.man_valid)
#According to four of the seven indices the best method is the Hierarchical one with two clusters.

#COMPLETE 
com.euc_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
                       metric="euclidean", method="complete")
summary(com.euc_valid)
#According to three of the seven indices the best method is the Hierarchical one with two clusters.


com.man_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
                       metric="manhattan", method="complete")
summary(com.man_valid)
#According to three of the seven indices the best method is the K-means with two clusters.

#SINGLE 
sin.euc_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
                       metric="euclidean", method="single")
summary(sin.euc_valid)
#According to four of the seven indices the best method is the Hierarchical one with two clusters.


sin.man_valid<-clValid(scale_df, nClust=2:7, clMethods=clmethod, validation=c("internal","stability"), maxitems=600,
                       metric="manhattan", method="single")
summary(sin.man_valid)
#According to four of the seven indices the best method is the Hierarchical one with two clusters.

#WARD
ward.euc_valid<-clValid(scale_df, nClust=2:7, clMethods="hierarchical", validation=c("internal","stability"), maxitems=600,
                       metric="euclidean", method="ward")
summary(ward.euc_valid)
ward.man_valid<-clValid(scale_df, nClust=2:7, clMethods="hierarchical", validation=c("internal","stability"), maxitems=600,
                        metric="manhattan", method="ward")
summary(ward.man_valid)

##The conclusion is that the Hierarchical with k=2 clusters is the best clustering method.
