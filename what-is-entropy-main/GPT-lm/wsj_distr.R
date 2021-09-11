require("data.table")
require("ggplot2")

data = fread("wsj_gpt2_CrossEntropy.txt")

p = ggplot(data, aes(x=V1)) + geom_density()

pdf("wsj_gpt2_CrossEntropy_density.pdf", 5, 5)
plot(p)
dev.off()


# Distribution of sentence lengths
d2 = fread("../data/wsj_raw_utf8_lwc.txt")
p = ggplot(d2, aes(x=V1)) + geom_density()
pdf("wsj_lengths_density.pdf", 5, 5)
plot(p)
dev.off()