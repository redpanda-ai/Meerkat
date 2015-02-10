#library(YodleeInsightsConnector)
#conn <- YodleeInsightsConnector.connect("askul", "@SkU1Y0dl33")

#query.table.clean     <- 'yi_stage_clean.test_framework_01'
#query.table.panel     <- 'yi_base_views.bank_panel'
#query           <- paste("select description from", query.table.clean)
#query.result    <- YodleeInsightsConnector.query(conn, query)

#library(tm)
#library(tm.plugin.dc)
#corpus <- as.DistributedCorpus(VCorpus(VectorSource(data.source)))
#corpus <- tm_map(as.Corpus(corpus), removeWords, stopwords("english"))

# LOAD THE DATA FROM S3
data.source <- read.table(file='/home/ubuntu/test_framework_03.txt000.gz', sep='|', quote="", comment.char="", numerals="no.loss")
colnames(data.source) <- c('transaction_id', 'description', 'merchant', 'random')
data.source$description <- as.character(data.source$description)

# PRE-PROCESSING: REMOVE STOP WORDS, MULTI-SPACES
library(tau)
words <- readLines(system.file("stopwords", "english.dat", package = "tm"))
data.removed <- lapply(data.source$description, remove_stopwords, words, lines=T)
data.source$clean_description <- sapply(data.removed, paste0, collapse=',')
data.source$clean_description <- gsub('[[:space:]]+', ' ', data.source$clean_description)

# TOKEN GENERALIZATION
library(qdap)
prep0 <- function(pattern, replacement, x) {
  print(pattern)
  return (gsub(pattern, replacement, x))
}

prep <- function(pattern, replacement, x) {
  print(pattern)
  matches           <- gregexpr(pattern, x)
  print(any(unlist(matches) != -1))
  if(any(unlist(matches) != -1)) {
    matches.strings   <- regmatches(x, matches)
    remove('matches')
    isreplace         <- lapply(matches.strings, '%in%', valid_words)
    strings.toreplace <- unique(unlist(matches.strings)[unlist(isreplace)==FALSE])
    remove('matches.strings')
    cat("Number of strings to replace : ", length(strings.toreplace), "\n")
    index.toreplace   <- grepl(pattern, x) & unlist(lapply(isreplace, function(x) any(x==FALSE)))
    cat("Number of indices : ", sum(index.toreplace==TRUE), "\n")
    if(length(strings.toreplace) > 0){
      for(i in 0:floor(length(strings.toreplace)/2500)){
        cat("Running iter : ", i, "\n")
        max <- min((i+1)*2500, length(strings.toreplace))
        x[index.toreplace==TRUE]     <- mgsub(strings.toreplace[i*2500+1:max], replacement, x[index.toreplace==TRUE])
      }
    }
  }
  return (x)
}

preplace <- function(x, pattern, replacement) {
  match     <- gregexpr(pattern, x)
  if(any(unlist(match) != -1)) {
    match.strings     <- unlist(regmatches(x, match))
    isreplace         <- !(match.strings %in% valid_words)
    for(word in match.strings[which(isreplace == TRUE)]) 
      x <- gsub(word, replacement, x)
  }
  x
}

mysub   <- function(pattern, replacement, lookup, x, ...) {
  if (length(pattern)!=length(replacement)) {
    stop("pattern and replacement do not have the same length.")
  }
  result <- x
  for (i in 1:length(pattern)) {
    pat       <- paste0("\\<", pattern[i], "\\>")
    if(lookup[i] == 0)
      result    <- prep0(pat, replacement[i], result)
    else
      result    <- prep(pat, replacement[i], result)
  }
  result  
}

mgsub_old <- function(pattern, replacement, x, ...) {
  if (length(pattern)!=length(replacement)) {
    stop("pattern and replacement do not have the same length.")
  }
  result <- x
  for (i in 1:length(pattern)) {
    pat               <- paste0("\\<", pattern[i], "\\>")
    print(paste("checking for", pat))
    #pat.match         <- regexpr(pattern[i], result)
    pat.match         <- regexpr(pat, result)
    print(pat.match)
    if(length(which(pat.match != -1)) > 0)  {
      pat.match.strings <- rep("", length(pat.match))
      pat.match.strings[which(pat.match!=-1)] <- regmatches(result, pat.match)
      print(pat.match.strings)
      pat.isreplace     <- grepl(pat, result) & !(pat.match.strings %in% valid_words)
      print(pat.isreplace)
      result[which(pat.isreplace==TRUE)] <- gsub(pat, replacement[i], result[which(pat.isreplace==TRUE)], ...)
      print(result)
    }
  }
  result
}
words.freq.corpus       <- read.csv('freq_words_corpus.txt', header = F, stringsAsFactors = F)
words.freq.tf           <- read.csv('freq-words-testframework.txt', header = F, stringsAsFactors = F)
words.merchant          <- read.csv('merchant_words.txt', header = T, stringsAsFactors = F)
word.classes            <- read.csv('word_classes.csv', header = T)
valid_words             <<- c(words.freq.corpus$V2, words.freq.tf$V2, as.character(words.merchant$Word), as.character(word.classes$replace))

data.source$generalized_description <- mysub(word.classes$pattern, word.classes$replace, word.classes$status, data.source$clean_description)

#GEO Generalization
library(maps)
library(stringr)
data(us.cities)
us.states <<- unique(us.cities$country.etc)
georep <- function(x) {
  matches <- regexpr(" [[:alpha:]]{2}$", x)
  strings <- rep(" ", length(x))
  strings[matches!=-1] <- regmatches(x, matches)
  isreplace <- str_trim(strings) %in% us.states
  x[isreplace == TRUE] <- gsub("[[:alpha:]]{2}$", "STATE", x[isreplace==TRUE])
  x
}

data.source$geo_description <- georep(data.source$generalized_description)
write.table(data.source, file='test_framework_01_generalized.txt', sep = '|', quote = F)
