library(data.table)
library(ggplot2)


# Read all _cross_entropy.txt files and combine them
all_ce_files = Sys.glob("../data/maptask/*_gpt_cross_entropy.txt") # 128 files in total
d_ce_list = list()
for (i in 1:length(all_ce_files)) {
    fname = all_ce_files[i]
    d = fread(fname, sep=",")
    setnames(d, c("utterIdx", "wordIdx", "crossEntropy"))
    d$dialogID = i
    d_ce_list[[i]] = d
}
d_ce = rbindlist(d_ce_list)

# CE at wordIdx == 0
mean(d_ce[wordIdx==0]$crossEntropy) # 8.50
sd(d_ce[wordIdx==0]$crossEntropy) # 0.64
# CE at wordIdx == 1
mean(d_ce[wordIdx==1]$crossEntropy) # 6.80
sd(d_ce[wordIdx==1]$crossEntropy) # 1.85
# CE at wordIdx == 2
mean(d_ce[wordIdx==2]$crossEntropy) # 8.67
sd(d_ce[wordIdx==2]$crossEntropy) # 1.81


# crossEntropy vs wordIdx
p = ggplot(d_ce[wordIdx<=50], aes(x=wordIdx, y=crossEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

# crossEntropy vs utterIdx
d_ce_utterMean = d_ce[, .(meanCrossEntropy = mean(crossEntropy)), by = .(dialogID, utterIdx)]
p1 = ggplot(d_ce_utterMean[utterIdx<=200], aes(x=utterIdx, y=meanCrossEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

p2 = ggplot(d_ce_utterMean[utterIdx<=10], aes(x=utterIdx, y=meanCrossEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

summary(lm(meanCrossEntropy ~ utterIdx, d_ce_utterMean[utterIdx<=200]))



###
# Read all *_entropy.txt files and combine them
all_e_files = Sys.glob("../data/maptask/*_gpt_entropy.txt")
d_e_list = list()
for (i in 1:length(all_e_files)) {
    fname = all_e_files[i]
    d = fread(fname, sep=",")
    setnames(d, c("utterIdx", "wordIdx", "entropy"))
    d$dialogID = i
    d_e_list[[i]] = d
}
d_e = rbindlist(d_e_list)


# entropy vs wordIdx
pp = ggplot(d_e[wordIdx<=50], aes(x=wordIdx, y=entropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

# entropy vs utterIdx
d_e_utterMean = d_e[, .(meanEntropy = mean(entropy)), by = .(dialogID, utterIdx)]
pp1 = ggplot(d_e_utterMean[utterIdx<=200], aes(x=utterIdx, y=meanEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

pp2 = ggplot(d_e_utterMean[utterIdx<=10], aes(x=utterIdx, y=meanEntropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='errorbar') +
    stat_summary(fun.y=mean, geom='line')

summary(lm(meanEntropy ~ utterIdx, d_e_utterMean[utterIdx<=200 & utterIdx>=10]))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)  3.6896115  0.0216761 170.215   <2e-16 ***
# utterIdx    -0.0002308  0.0002022  -1.142    0.254
