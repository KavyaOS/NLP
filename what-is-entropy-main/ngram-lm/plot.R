library(data.table)
library(ggplot2)


# plot unigram entropy vs. word position in sentence
data = fread('wsj_raw_repl_lower.txt.ent_in_sent')
setnames(data, c('sentId', 'wordPos', 'entropy'))
data = data[entropy>0.0]

# remove outliers in wordPos
wordPos_mean = mean(data$wordPos)
wordPos_sd = sd(data$wordPos)
data = data[wordPos <= wordPos_mean + 2*wordPos_sd]

p = ggplot(data, aes(x=wordPos, y=entropy)) + stat_smooth()

p = ggplot(data, aes(x=wordPos, y=entropy)) +
    stat_summary(fun.data=mean_cl_boot, geom='ribbon', alpha=0.5) +
    stat_summary(fun.y=mean, geom='line')
pdf('figs/wsj_raw_repl_lower.unigram.entropy_in_sentence.pdf', 5, 5)
plot(p)
dev.off()


# plot entropy and surprisal together
data2 = fread('wsj_raw_repl_lower.txt.surp_in_sent')
setnames(data2, c('sentId', 'wordPos', 'surprisal'))
data2 = data2[wordPos <= wordPos_mean + 2*wordPos_sd]

# combine
data3 = rbindlist(list(data, data2))
setnames(data3, 'entropy', 'value')
data3$type = 'surprisal'
data3[1:nrow(data),]$type = 'entropy'

p = ggplot(data3, aes(x=wordPos, y=value)) +
    stat_summary(fun.data=mean_cl_boot, geom='ribbon', alpha=0.5, aes(fill=type)) +
    stat_summary(fun.y=mean, geom='line', aes(lty=type))
pdf('figs/wsj_raw_repl_lower.unigram.entropy_surprisal_in_sentence.pdf', 5, 5)
plot(p)
dev.off()


# Examine the distributions of entropy vs. surprisal
p = ggplot(data3, aes(value)) + geom_density(aes(fill=type), alpha=.5)
pdf('figs/wsj_raw_repl_lower.unigram.entropy_surprisal_distributions.pdf', 5, 5)
plot(p)
dev.off()
