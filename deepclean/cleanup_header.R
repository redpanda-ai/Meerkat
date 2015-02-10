preprocess <- function(x, stop_words) {
  data.removed <- lapply(x$description, remove_stopwords, stop_words, lines=T)
  data.clean <- sapply(data.removed, paste0, collapse=',')
  data.clean <- gsub('[[:space:]]+', ' ', data.clean)
}

mgsub_div <- function(x, replace.strings, replacement) {
  limit <- 100
  if(length(replace.strings) > 0){
    replace.strings <- unlist(lapply(replace.strings, function(x) paste0("\\<", x, "\\>")))
    for(i in 0:floor(length(replace.strings)/limit)){
      cat("Running iter : ", i, "\n")
      max   <- min((i+1)*limit, length(replace.strings))
      x     <- mgsub(replace.strings[i*limit+1:max], replacement, x)
    }
  }
  x
}

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
    isreplace         <- lapply(matches.strings, '%in%', valid_words)
    strings.toreplace <- unique(unlist(matches.strings)[unlist(isreplace)==FALSE])
    remove('matches.strings')
    remove('matches')
    cat("Number of strings to replace : ", length(strings.toreplace), "\n")
    index.toreplace   <- grepl(pattern, x) & unlist(lapply(isreplace, function(x) any(x==FALSE)))
    cat("Number of indices : ", sum(index.toreplace==TRUE), "\n")
    x[index.toreplace==TRUE] <- mgsub_div(x[index.toreplace==TRUE], strings.toreplace, replacement)
  }
  return (x)
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

georep <- function(x) {
  matches <- regexpr(" [[:alpha:]]{2}$", x)
  strings <- rep(" ", length(x))
  strings[matches!=-1] <- regmatches(x, matches)
  isreplace <- str_trim(strings) %in% us.states
  x[isreplace == TRUE] <- gsub("[[:alpha:]]{2}$", "STATE", x[isreplace==TRUE])
  x
}

cityrep <- function(x, list.cities, list.merchants) {
  #data.words    <- unlist(lapply(x, strsplit, "[[:space:]]", perl=T))
  #words.replace <- unique(unlist(lapply(list.cities, agrep, data.words, value=T, max.distance=1)))
  x <- mgsub_div(x, list.cities, "LOC")
  x
}
