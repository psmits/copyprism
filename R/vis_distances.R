library(ggplot2)
library(dplyr)
library(readr)
library(magrittr)


# bring in validation information
valid_results <- read_csv('../results/text_comparison.csv')

# similarity between reference text and generated
# similarity between reference text and random
# difference in similarity
# positive difference == generated better than random
# negative difference == random better than generated
# difference of 1 means generated is effectively reference
# range is [-1, 1]

# text was encoded as tf-idf embeddings
# cosine similarty 

# pretty histogram of difference in cosine similarity between 

tt <- 
  valid_results %>%
  ggplot(aes(x = distance)) +
  geom_histogram(fill = 'goldenrod') +
  theme_light()

ggsave(filename = '../results/distance_dist.png', 
       plot = tt, 
       width = 8, 
       height = 6)
