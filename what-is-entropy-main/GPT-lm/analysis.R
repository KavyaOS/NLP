library(data.table)
library(ggplot2)

d1 = fread("./wsj_raw_utf8.txt.GPT_cross_entropy.column")
setnames(d1, c("sentId", "wordPos", "crossEntropy"))

summary(lm(crossEntropy ~ wordPos, d1))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept) 9.2262440  0.0105050 878.274   <2e-16 ***
# wordPos     0.0032524  0.0003873   8.398   <2e-16 ***
# NOTE: it conforms with previous findings

d1 = d1[complete.cases(d1),]

p = ggplot(scale(d1), aes(crossEntropy)) + geom_density()
pdf("GPT_crossEntropy.pdf", 5, 5)
plot(p)
dev.off()

# How CE increases with wordPos
p = ggplot(d1[wordPos<=50], aes(x=wordPos, y=crossEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')
pdf("GPT_crossEntropy_wordPos.pdf", 10, 5)
plot(p)
dev.off()

# Normality test
shapiro.test(sample(d1$crossEntropy, 5000))
# Shapiro-Wilk normality test
# data:  sample(data$crossEntropy, 5000)
# W = 0.97934, p-value < 2.2e-16
shapiro.test(sample(
    scale(d1$crossEntropy), 5000
))


# Entropy
d2 = fread("../ngram-lm/wsj_raw_utf8.txt.GPT_entropy")
setnames(d2, c("sentId", "wordPos", "entropy"))

summary(lm(entropy ~ wordPos, d2))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)  4.2801811  0.0080314  532.93   <2e-16 ***
# wordPos     -0.0141975  0.0002961  -47.95   <2e-16 ***
# NOTE: entropy decreases

p = ggplot(d2, aes(entropy)) + geom_density()
pdf("GPT_entropy_density.pdf", 5, 5)
plot(p)
dev.off()

p = ggplot(d2[wordPos<=50], aes(x=wordPos, y=entropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')
pdf("GPT_entropy_wordPos.pdf", 10, 5)
plot(p)
dev.off()


# Normality test
shapiro.test(sample(d2$entropy, 5000))
# Shapiro-Wilk normality test
# data:  sample(d2$entropy, 5000)
# W = 0.99181, p-value < 2.2e-16


## Combine crossEntropy and entropy data
setkey(d1, sentId, wordPos)
setkey(d2, sentId, wordPos)
d3 = d1[d2, nomatch=0]

d4 = melt(d3, id = c("sentId", "wordPos"), variable.name="type")

p = ggplot(d4[wordPos<=50], aes(x=wordPos, y=value, linetype=type, color=type)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

# correlation
cor.test(d3$crossEntropy, d3$entropy)
# t = -105.84, df = 102380, p-value < 2.2e-16
